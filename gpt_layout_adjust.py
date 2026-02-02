"""Generate and manually adjust GPT task layouts via object_layout_task.

Workflow:
1) Generate a GPT task (writes JSON/PKL/PY via scripts/advanced/gpt_gen.py code).
2) Launch get_started/obj_layout/object_layout_task.py for manual layout tweaking.
3) After saving poses (press C) and exiting, convert saved poses into:
   - metasim/cfg/tasks/gpt/config/tasks/<task>.json
   - roboverse_data/trajs/gpt/<task>/franka_v2.pkl (init_state updated)
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from loguru import logger as log

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Args:
    # Task selection
    task: str | None = None  # e.g. "gpt.organizecondimentsintobasket"

    # Optional GPT generation (requires OPENAI_API_KEY + network)
    prompt: str | None = None
    difficulty: int = 3
    human_check: bool = False
    allow_table: bool = False
    """Allow GPT to select a `table` asset (enables tabletop mode when selected)."""

    # Simulator options (forwarded to object_layout_task.py)
    sim: str = "mujoco"
    renderer: str | None = None
    robot: str = "franka"
    scene: str | None = None
    num_envs: int = 1
    headless: bool = False
    enable_viser: bool = False
    viser_port: int = 8080
    display_camera: bool = False

    # Where object_layout_task writes pose snapshots
    poses_output_dir: str = "get_started/output"
    poses_output_basename: str = "saved_poses"

    # Apply an existing saved poses file (skip launching the UI)
    poses_path: str | None = None
    apply_only: bool = False


TASK_JSON_DIR = Path("metasim/cfg/tasks/gpt/config/tasks")
TASK_PKL_DIR = Path("roboverse_data/trajs/gpt")


def _normalize_task_name(name: str) -> tuple[str, str]:
    """Return (full_task_name, snake_name)."""
    raw = name.strip()
    if raw.lower().startswith("gpt."):
        snake = raw.split(".", 1)[1]
        return f"gpt.{snake}", snake
    if raw.lower().startswith("gpt:"):
        snake = raw.split(":", 1)[1].strip().lower()
        return f"gpt.{snake}", snake
    # Assume snake
    snake = raw.strip().lower()
    return f"gpt.{snake}", snake


def _load_saved_poses_py(path: Path) -> dict[str, Any]:
    """Load `poses` dict from a saved_poses_*.py file."""
    spec = importlib.util.spec_from_file_location("roboverse_saved_poses", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import poses module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    poses = getattr(module, "poses", None)
    if not isinstance(poses, dict):
        raise ValueError(f"Expected `poses` dict in {path}, got {type(poses)!r}")
    return poses


def _tensor_like_to_list(x: Any) -> list[float]:
    # Supports torch.Tensor and numpy arrays without importing them directly.
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float(v) for v in x]

def _maybe_align_robot_z_for_tabletop(poses: dict[str, Any]) -> float | None:
    """If poses look like a tabletop scene, align each robot root z to the lowest object root z.

    Returns:
        The z used for alignment if applied, else None.
    """
    tabletop_robot_xy = (-0.5, 0.0)
    objects = poses.get("objects") or {}
    robots = poses.get("robots") or {}
    if not isinstance(objects, dict) or not isinstance(robots, dict) or not objects or not robots:
        return None

    # Identify likely support object(s) to exclude from the min-z computation.
    support_names: set[str] = set()
    if "table" in objects:
        support_names.add("table")
    else:
        for name in objects.keys():
            lname = str(name).lower()
            if lname in {"desk", "counter", "workbench"}:
                support_names.add(str(name))
            elif any(tok in lname for tok in ("table", "desk", "counter", "workbench")):
                support_names.add(str(name))

    z_by_name: list[tuple[str, float]] = []
    for name, entry in objects.items():
        if not isinstance(entry, dict) or "pos" not in entry:
            continue
        try:
            z = float(_tensor_like_to_list(entry["pos"])[2])
        except Exception:
            continue
        z_by_name.append((str(name), z))

    if not z_by_name:
        return None

    # Heuristic fallback: if no support object spotted by name but there is a clear "lowest" object,
    # treat it as support and exclude it.
    if not support_names and len(z_by_name) >= 2:
        z_sorted = sorted(z_by_name, key=lambda kv: kv[1])
        (min_name, min_z), (_, second_z) = z_sorted[0], z_sorted[1]
        if (second_z - min_z) > 0.15:  # 15cm gap is typical for table base vs tabletop objects
            support_names.add(min_name)

    min_obj_z: float | None = None
    for name, z in z_by_name:
        if name in support_names:
            continue
        if min_obj_z is None or z < min_obj_z:
            min_obj_z = z
    if min_obj_z is None:
        return None

    applied = False
    for rname, entry in robots.items():
        if not isinstance(entry, dict) or "pos" not in entry:
            continue
        try:
            pos = _tensor_like_to_list(entry["pos"])
        except Exception:
            continue
        if len(pos) != 3:
            continue
        old = (float(pos[0]), float(pos[1]), float(pos[2]))
        pos[0] = float(tabletop_robot_xy[0])
        pos[1] = float(tabletop_robot_xy[1])
        pos[2] = float(min_obj_z)
        entry["pos"] = pos
        applied = applied or any(abs(a - b) > 1e-6 for a, b in zip(old, pos, strict=False))

    return float(min_obj_z) if applied else None


def _poses_to_init_state(poses: dict[str, Any]) -> dict[str, Any]:
    init_state: dict[str, Any] = {}

    for name, entry in (poses.get("objects") or {}).items():
        init_state[name] = {
            "pos": _tensor_like_to_list(entry["pos"]),
            "rot": _tensor_like_to_list(entry["rot"]),
        }

    for name, entry in (poses.get("robots") or {}).items():
        robot_state = {
            "pos": _tensor_like_to_list(entry["pos"]),
            "rot": _tensor_like_to_list(entry["rot"]),
        }
        if "dof_pos" in entry and entry["dof_pos"] is not None:
            robot_state["dof_pos"] = {str(k): float(v) for k, v in dict(entry["dof_pos"]).items()}
        init_state[name] = robot_state

    return init_state


def _update_task_json(task_snake: str, poses: dict[str, Any]) -> Path:
    path = TASK_JSON_DIR / f"{task_snake}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        TASK_JSON_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "task_name": task_snake,
            "task_language_instruction": "",
            "objects_involved": sorted(list((poses.get("objects") or {}).keys())),
            "objects": [],
            "robots": [],
        }
    init_state = _poses_to_init_state(poses)

    obj_by_name = {o["name"]: o for o in data.get("objects", []) if isinstance(o, dict) and "name" in o}
    for name, entry in (poses.get("objects") or {}).items():
        dst = obj_by_name.get(name, {"name": name})
        dst["pos"] = init_state[name]["pos"]
        dst["rot"] = init_state[name]["rot"]
        obj_by_name[name] = dst
    data["objects"] = list(obj_by_name.values())

    robot_by_name = {r["name"]: r for r in data.get("robots", []) if isinstance(r, dict) and "name" in r}
    for name, entry in (poses.get("robots") or {}).items():
        dst = robot_by_name.get(name, {"name": name})
        dst["pos"] = init_state[name]["pos"]
        dst["rot"] = init_state[name]["rot"]
        if "dof_pos" in init_state[name]:
            dst["dof_pos"] = init_state[name]["dof_pos"]
        robot_by_name[name] = dst
    data["robots"] = list(robot_by_name.values())

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _update_task_pkl(task_snake: str, poses: dict[str, Any]) -> Path:
    path = TASK_PKL_DIR / task_snake / "franka_v2.pkl"
    init_state_updates = _poses_to_init_state(poses)
    if path.exists():
        with open(path, "rb") as f:
            traj_data = pickle.load(f)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        robots = poses.get("robots") or {}
        if not robots:
            robots = {"franka": {"dof_pos": {}}}
        traj_data = {}
        for robot_name, robot_entry in robots.items():
            dof_pos = dict(robot_entry.get("dof_pos") or {})
            zero_dof = {str(k): 0.0 for k in dof_pos.keys()}
            traj_data[robot_name] = [
                {"actions": [{"dof_pos_target": zero_dof}], "init_state": init_state_updates, "states": [], "extra": None}
            ]

    if not isinstance(traj_data, dict):
        raise ValueError(f"Unexpected PKL format in {path}: {type(traj_data)!r}")

    for robot_name, entries in traj_data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            existing_init_state = entry.get("init_state") or {}
            if not isinstance(existing_init_state, dict):
                existing_init_state = {}
            existing_init_state.update(init_state_updates)
            entry["init_state"] = existing_init_state

    with open(path, "wb") as f:
        pickle.dump(traj_data, f)

    return path


def _find_latest_saved_poses(dir_path: Path, basename: str) -> Path | None:
    if not dir_path.exists():
        return None
    candidates = sorted(dir_path.glob(f"{basename}_*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _sanitize_task_name_for_filename(task_name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in task_name.strip().lower())


def _find_latest_saved_poses_for_task(dir_path: Path, basename: str, task_name: str) -> Path | None:
    safe_task = _sanitize_task_name_for_filename(task_name)
    return _find_latest_saved_poses(dir_path, f"{basename}_{safe_task}")


def _maybe_generate_task(prompt: str, difficulty: int, *, allow_table: bool) -> tuple[str, str]:
    """Generate a GPT task and return (full_task_name, snake_name, json_path, pkl_path, py_path)."""
    try:
        from scripts.advanced.gpt_gen import GPTConfig, PathConfig, TaskGenerator, TaskWriter
        import openai  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GPT generation requires `openai` and scripts/advanced/gpt_gen.py dependencies to be importable."
        ) from e

    from scripts.advanced.gpt_gen import GPTConfig, PathConfig, TaskGenerator, TaskWriter
    import openai

    gpt_cfg = GPTConfig.from_env(difficulty=difficulty)
    if not gpt_cfg.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for --prompt")

    client = openai.OpenAI(api_key=gpt_cfg.api_key, base_url=gpt_cfg.base_url)
    paths = PathConfig()
    generator = TaskGenerator(client, paths)
    writer = TaskWriter(paths)

    # Pass policy via env var to gpt_gen internals (used by Stage1/2 prompts).
    os.environ["TASKGEN_ALLOW_TABLE"] = "1" if allow_table else "0"
    spec, reasoning_metadata = generator.generate(prompt, difficulty)
    if spec is None:
        stage3_error = (reasoning_metadata or {}).get("stage3_error")
        if isinstance(stage3_error, dict) and stage3_error.get("error_type") == "validation_failed":
            first = ((stage3_error.get("physics_report") or {}).get("errors") or [{}])[0]
            raise RuntimeError(f"Layout validation failed. First error: {first}")
        stage2_failures = (reasoning_metadata or {}).get("stage2_failures")
        stage2_raw = (reasoning_metadata or {}).get("stage2_raw")
        if stage2_failures or stage2_raw:
            msg = "GPT generation returned None.\n"
            if stage2_failures:
                msg += f"Stage2 failures: {stage2_failures}\n"
            if stage2_raw:
                preview = stage2_raw if len(stage2_raw) <= 2000 else (stage2_raw[:2000] + "\n...[truncated]...")
                msg += f"Stage2 raw (preview):\n{preview}\n"
            msg += "Try a more specific prompt, or reduce ambiguity (objects, actions, constraints)."
            raise RuntimeError(msg)
        raise RuntimeError("GPT generation returned None; try a more specific prompt.")

    json_path, pkl_path, py_path = writer.write_all(spec, reasoning_metadata)
    full_task, snake = _normalize_task_name(f"gpt.{spec.snake_name}")
    return full_task, snake, json_path, pkl_path, py_path


def _delete_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except Exception as e:
            log.warning(f"Failed to delete {p}: {e}")


def _ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return default_yes
    if not ans:
        return default_yes
    if ans in {"y", "yes"}:
        return True
    if ans in {"n", "no"}:
        return False
    return default_yes


def _launch_object_layout(args: Args, full_task_name: str) -> None:
    cmd = [
        sys.executable,
        "get_started/obj_layout/object_layout_task.py",
        "--task",
        full_task_name,
        "--robot",
        args.robot,
        "--sim",
        args.sim,
        "--num-envs",
        str(args.num_envs),
        "--viser-port",
        str(args.viser_port),
        "--poses-output-dir",
        args.poses_output_dir,
        "--poses-output-basename",
        args.poses_output_basename,
    ]
    cmd += ["--headless"] if args.headless else ["--no-headless"]
    if args.scene is not None:
        cmd += ["--scene", args.scene]
    if args.renderer is not None:
        cmd += ["--renderer", args.renderer]
    if args.enable_viser:
        cmd += ["--enable-viser"]
    else:
        cmd += ["--no-enable-viser"]
    if args.display_camera:
        cmd += ["--display-camera"]
    else:
        cmd += ["--no-display-camera"]

    log.info("Launching layout editor:")
    log.info(" ".join(cmd))
    subprocess.run(cmd, check=False)


def main() -> None:
    args = tyro.cli(Args)
    session_start_time = time.time()

    generated_paths: dict[str, Path] | None = None

    # Generation path (aligns with scripts/advanced/gpt_gen.py):
    # - If --prompt is provided, generate exactly one task from that prompt.
    # - If neither --task nor --prompt is provided, prompt interactively for the same inputs as gpt_gen.
    if args.prompt is not None:
        full_task_name, task_snake, json_path, pkl_path, py_path = _maybe_generate_task(
            args.prompt, args.difficulty, allow_table=args.allow_table
        )
        generated_paths = {"json": Path(json_path), "pkl": Path(pkl_path), "py": Path(py_path)}
    elif args.task is not None:
        full_task_name, task_snake = _normalize_task_name(args.task)
    else:
        from scripts.advanced.gpt_gen import TaskGenUI

        prompt = TaskGenUI.get_user_prompt()
        difficulty = TaskGenUI.get_difficulty()
        full_task_name, task_snake, json_path, pkl_path, py_path = _maybe_generate_task(
            prompt, difficulty, allow_table=args.allow_table
        )
        generated_paths = {"json": Path(json_path), "pkl": Path(pkl_path), "py": Path(py_path)}

    poses_dir = Path(args.poses_output_dir)
    poses_path = Path(args.poses_path) if args.poses_path else None

    if args.apply_only:
        if poses_path is None:
            poses_path = _find_latest_saved_poses(poses_dir, args.poses_output_basename)
        if poses_path is None:
            raise FileNotFoundError(f"No saved poses found in: {poses_dir} (basename={args.poses_output_basename})")
        poses = _load_saved_poses_py(poses_path)
        aligned_z = _maybe_align_robot_z_for_tabletop(poses)
        if aligned_z is not None:
            log.info(f"Tabletop: set robot xy=(-0.5,0.0) and aligned z to lowest object z={aligned_z:.6f} (excluding support).")
        json_path = _update_task_json(task_snake, poses)
        pkl_path = _update_task_pkl(task_snake, poses)
        log.info(f"Updated task JSON: {json_path}")
        log.info(f"Updated task PKL:  {pkl_path}")
        return

    if not args.human_check:
        log.info("Human check disabled; skipping layout adjustment.")
        log.info("Replay demo command:")
        log.info(f"  python scripts/advanced/replay_demo.py --sim=mujoco --task=gpt.{task_snake} --num_envs 1 --headless")
        return

    _launch_object_layout(args, full_task_name)

    used_poses_path = _find_latest_saved_poses_for_task(poses_dir, args.poses_output_basename, full_task_name)
    if used_poses_path is None:
        log.warning(f"No saved poses found in {poses_dir}; press C in the UI to save poses before exiting.")
        if generated_paths is not None:
            keep_task = _ask_yes_no("Keep this generated task? [Y/n]: ", default_yes=True)
            if not keep_task:
                log.info("Discarding generated task (no saved poses).")
                _delete_paths([generated_paths["json"], generated_paths["py"], generated_paths["pkl"].parent])
                return

        log.info("Replay demo command:")
        log.info(f"  python scripts/advanced/replay_demo.py --sim=mujoco --task=gpt.{task_snake} --num_envs 1 --headless")
        return

    poses_to_delete: list[Path] = []
    try:
        safe_task = _sanitize_task_name_for_filename(full_task_name)
        for p in poses_dir.glob(f"{args.poses_output_basename}_{safe_task}_*.py"):
            if p.stat().st_mtime >= session_start_time:
                poses_to_delete.append(p)
    except Exception as e:
        log.warning(f"Failed to scan poses in {poses_dir}: {e}")

    if generated_paths is not None:
        keep_task = _ask_yes_no("Keep this generated task? [Y/n]: ", default_yes=True)
        if not keep_task:
            log.info("Discarding generated task and saved poses from this session.")
            _delete_paths([generated_paths["json"], generated_paths["py"], generated_paths["pkl"].parent, *poses_to_delete])
            return
    else:
        keep_changes = _ask_yes_no("Keep these layout changes? [Y/n]: ", default_yes=True)
        if not keep_changes:
            log.info("Discarding layout changes (leaving task files unchanged).")
            _delete_paths(poses_to_delete)
            return

    poses = _load_saved_poses_py(used_poses_path)
    aligned_z = _maybe_align_robot_z_for_tabletop(poses)
    if aligned_z is not None:
        log.info(f"Tabletop: set robot xy=(-0.5,0.0) and aligned z to lowest object z={aligned_z:.6f} (excluding support).")
    json_path = _update_task_json(task_snake, poses)
    pkl_path = _update_task_pkl(task_snake, poses)
    log.info(f"Saved poses:       {used_poses_path}")
    log.info(f"Updated task JSON: {json_path}")
    log.info(f"Updated task PKL:  {pkl_path}")
    log.info("Replay demo command:")
    log.info(f"  python scripts/advanced/replay_demo.py --sim=mujoco --task=gpt.{task_snake} --num_envs 1 --headless")
    return

if __name__ == "__main__":
    main()
