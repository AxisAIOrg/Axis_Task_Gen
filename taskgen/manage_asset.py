#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import openai
import tyro
from loguru import logger as log

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKGEN_DIR = _REPO_ROOT / "taskgen_json"

_RIGID_REGISTRY = _TASKGEN_DIR / "category_registry_objects.json"
_ART_REGISTRY = _TASKGEN_DIR / "category_registry_articulated_objects.json"


ObjectType = Literal["rigid", "articulated"]


@dataclass
class Args:
    asset_root: str
    """Asset root directory that contains multiple asset folders (each folder name is an asset name).

    Example:
      roboverse_data/assets/libero/COMMON/stable_hope_objects
    """

    dry_run: bool = False
    limit: int | None = None
    only: str | None = None
    """If set, process only this single asset name (exact match)."""

    object_type: ObjectType = "rigid"
    """All assets under this asset_root are treated as the same type.

    One asset_root should contain either all rigid objects or all articulated objects.
    """

    overwrite: Literal["ask", "yes", "no"] = "ask"
    """What to do if an asset already exists in taskgen_json registries.

    - ask: prompt per asset
    - yes: overwrite without prompting (delete old records then re-add)
    - no: skip existing assets
    """

    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))
    sleep_s: float = float(os.getenv("OPENAI_SLEEP_S", "0.0"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_json(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_json_response(content: str) -> dict[str, Any] | None:
    raw = _extract_json(content)

    def _strip_to_braces(x: str) -> str:
        x = x.strip()
        if x.startswith("{") and x.endswith("}"):
            return x
        i = x.find("{")
        j = x.rfind("}")
        if i != -1 and j != -1 and j > i:
            return x[i : j + 1]
        return x

    def _remove_trailing_commas(x: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", x)

    for cand in (
        raw,
        _strip_to_braces(raw),
        _remove_trailing_commas(raw),
        _remove_trailing_commas(_strip_to_braces(raw)),
    ):
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


def _norm_category_name(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _pick_first_file(folder: Path, exts: tuple[str, ...], prefer_stem: str | None = None) -> Path | None:
    if not folder.exists() or not folder.is_dir():
        return None
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    files.sort(key=lambda p: p.name.lower())
    if prefer_stem:
        prefer_stem_l = prefer_stem.lower()
        for p in files:
            if p.stem.lower() == prefer_stem_l:
                return p
        for p in files:
            if prefer_stem_l in p.name.lower():
                return p
    return files[0]


def _to_repo_rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(_REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _ensure_articulated_registry() -> None:
    if _ART_REGISTRY.exists():
        return
    template: dict[str, Any] = {
        "$schema": "RoboVerse Object Category Registry v2.0",
        "description": "Master registry for articulated object categories with references to detailed category files",
        "version": "2.0.0",
        "registry_type": "articulated_objects",
        "categories": {},
        "usage_notes": {
            "architecture": "Three-layer file architecture: 1) This master registry for Stage 1 LLM reasoning. 2) Object type folders (rigid/, articulated/, soft_body/) for organization. 3) Category detail files for Stage 2 LLM reasoning.",
            "llm_stage_1": "Use this file directly for semantic reasoning - contains all articulated category information without loading detailed object data",
            "llm_stage_2": "Load specific category detail files (via detail_file paths) only for selected categories to get articulated object specifications",
            "extensibility": "To add new objects: update the corresponding file in objects/articulated/, then update object_names and object_count in this file.",
        },
    }
    _write_json(_ART_REGISTRY, template)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")
    reg = _load_json(path)
    if not isinstance(reg.get("categories"), dict):
        raise ValueError(f"Invalid registry (missing categories dict): {path}")
    return reg


def _existing_categories_summary(rigid_reg: dict[str, Any], art_reg: dict[str, Any]) -> str:
    def _fmt(reg: dict[str, Any], label: str) -> str:
        cats = reg.get("categories") or {}
        lines = [f"{label} categories:"]
        if not cats:
            lines.append("- (none)")
            return "\n".join(lines)
        for k in sorted(cats.keys()):
            v = cats[k] or {}
            desc = str(v.get("category_description") or "").strip()
            objs = v.get("object_names") or []
            if not isinstance(objs, list):
                objs = []
            sample = ", ".join([str(x) for x in objs[:8]])
            lines.append(f"- {k}: {desc} (examples: {sample})")
        return "\n".join(lines)

    return _fmt(rigid_reg, "Rigid") + "\n\n" + _fmt(art_reg, "Articulated")


def _call_gpt(
    client: openai.OpenAI,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str,
    user: str,
) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    parsed = _parse_json_response(content)
    if parsed is None:
        preview = content if len(content) <= 4000 else (content[:4000] + "\n...[truncated]...")
        raise ValueError(f"Failed to parse JSON from model response:\n{preview}")
    return parsed


def _estimate_tabletop_from_mjcf(mjcf_path: Path, *, margin_xy: float = 0.03, top_band: float = 0.02) -> dict[str, Any] | None:
    try:
        import mujoco  # type: ignore
    except Exception:
        log.warning("mujoco is not available; cannot auto-estimate tabletop metadata.")
        return None
    if not mjcf_path.exists():
        return None
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as e:
        log.warning("Failed to load MJCF for tabletop estimate {}: {}", mjcf_path, e)
        return None
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    def _geom_top_z(gid: int) -> float:
        z0 = float(data.geom_xpos[gid][2])
        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return z0
        try:
            sx, sy, sz = float(model.geom_size[gid][0]), float(model.geom_size[gid][1]), float(model.geom_size[gid][2])
        except Exception:
            sx = sy = sz = 0.0
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            return z0 + sz
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return z0 + sx
        if gtype in {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}:
            return z0 + sy + sx
        if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return z0 + sz
        try:
            return z0 + float(model.geom_rbound[gid])
        except Exception:
            return z0

    geom_ids = [gid for gid in range(int(model.ngeom)) if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
    if not geom_ids:
        return None
    top_z = max(_geom_top_z(gid) for gid in geom_ids)
    band = max(0.0, float(top_band))
    top_gids = [gid for gid in geom_ids if _geom_top_z(gid) >= (top_z - band)]
    if not top_gids:
        top_gids = list(geom_ids)

    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")
    for gid in top_gids:
        cx, cy, _ = [float(v) for v in data.geom_xpos[gid]]
        gtype = int(model.geom_type[gid])
        ex = ey = 0.0
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            sx, sy, sz = [float(v) for v in model.geom_size[gid]]
            R = [float(v) for v in data.geom_xmat[gid]]
            r00, r01, r02, r10, r11, r12 = abs(R[0]), abs(R[1]), abs(R[2]), abs(R[3]), abs(R[4]), abs(R[5])
            ex = r00 * sx + r01 * sy + r02 * sz
            ey = r10 * sx + r11 * sy + r12 * sz
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            r = float(model.geom_size[gid][0])
            ex = ey = r
        else:
            r = float(getattr(model, "geom_rbound", [0.0] * int(model.ngeom))[gid])
            ex = ey = max(0.0, r)

        xmin = min(xmin, cx - ex)
        xmax = max(xmax, cx + ex)
        ymin = min(ymin, cy - ey)
        ymax = max(ymax, cy + ey)

    if not (xmin < xmax and ymin < ymax):
        return None
    m = max(0.0, float(margin_xy))
    return {
        "height": float(top_z),
        "range": {"xmin": float(xmin + m), "xmax": float(xmax - m), "ymin": float(ymin + m), "ymax": float(ymax - m)},
    }


def _validate_gpt_payload(payload: dict[str, Any], *, asset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cat = payload.get("category")
    if not isinstance(cat, dict):
        raise ValueError(f"{asset_name}: missing category object")
    cat_name = cat.get("name")
    if not isinstance(cat_name, str) or not cat_name.strip():
        raise ValueError(f"{asset_name}: category.name must be non-empty string")
    cat_name = _norm_category_name(cat_name)
    if not cat_name:
        raise ValueError(f"{asset_name}: category.name normalized to empty")
    cat["name"] = cat_name

    if not isinstance(cat.get("description"), str) or not str(cat.get("description")).strip():
        raise ValueError(f"{asset_name}: category.description must be non-empty string")

    common = cat.get("common_properties")
    if not isinstance(common, dict):
        raise ValueError(f"{asset_name}: category.common_properties must be an object")
    if not isinstance(common.get("typical_location"), str) or not str(common.get("typical_location")).strip():
        raise ValueError(f"{asset_name}: category.common_properties.typical_location must be non-empty string")
    if not isinstance(common.get("manipulation_type"), str) or not str(common.get("manipulation_type")).strip():
        raise ValueError(f"{asset_name}: category.common_properties.manipulation_type must be non-empty string")

    obj = payload.get("object")
    if not isinstance(obj, dict):
        raise ValueError(f"{asset_name}: missing object object")
    if not isinstance(obj.get("description"), str) or not str(obj.get("description")).strip():
        raise ValueError(f"{asset_name}: object.description must be non-empty string")

    meta = obj.get("metadata")
    if meta is None:
        meta = {}
        obj["metadata"] = meta
    if not isinstance(meta, dict):
        raise ValueError(f"{asset_name}: object.metadata must be an object")
    tags = meta.get("tags")
    if tags is None:
        meta["tags"] = []
    elif not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValueError(f"{asset_name}: object.metadata.tags must be a list of strings")

    # Optional init_state hints.
    init = obj.get("suggested_init_state")
    if init is not None and not isinstance(init, dict):
        raise ValueError(f"{asset_name}: object.suggested_init_state must be an object if provided")

    # Optional table-top metadata (script may also fill it automatically when category is table).
    tt = obj.get("tabletop")
    if tt is not None and not isinstance(tt, dict):
        raise ValueError(f"{asset_name}: object.tabletop must be an object if provided")

    return cat, obj


def _find_existing_asset(name: str, registry_paths: list[Path]) -> tuple[Path, str] | None:
    for reg_path in registry_paths:
        if not reg_path.exists():
            continue
        reg = _load_json(reg_path)
        cats = reg.get("categories") or {}
        if not isinstance(cats, dict):
            continue
        for cat_name, entry in cats.items():
            if not isinstance(entry, dict):
                continue
            names = entry.get("object_names")
            if isinstance(names, list) and name in names:
                return reg_path, str(cat_name)
            detail_file = entry.get("detail_file")
            if not isinstance(detail_file, str) or not detail_file:
                continue
            detail_path = _REPO_ROOT / detail_file if not Path(detail_file).is_absolute() else Path(detail_file)
            if not detail_path.exists():
                continue
            detail = _load_json(detail_path)
            objs = detail.get("objects") or []
            if not isinstance(objs, list):
                continue
            for o in objs:
                if isinstance(o, dict) and o.get("name") == name:
                    return reg_path, str(cat_name)
    return None


def _ask_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _remove_asset_from_registry(*, reg_path: Path, asset_name: str, dry_run: bool) -> bool:
    """Remove asset from a single registry and its referenced detail file.

    Returns True if something was removed.
    """
    if not reg_path.exists():
        return False
    reg = _load_json(reg_path)
    cats = reg.get("categories") or {}
    if not isinstance(cats, dict) or not cats:
        return False

    removed_any = False
    cats_to_delete: list[str] = []

    for cat_name, entry in list(cats.items()):
        if not isinstance(entry, dict):
            continue

        # Remove from registry list (even if detail file is missing).
        names = entry.get("object_names")
        if not isinstance(names, list):
            names = []
        orig_names = [str(n) for n in names if isinstance(n, str)]
        new_names = [n for n in orig_names if n != asset_name]
        if new_names != orig_names:
            removed_any = True
        entry["object_names"] = sorted(set(new_names))
        entry["object_count"] = len(entry["object_names"])

        detail_file = entry.get("detail_file")
        if isinstance(detail_file, str) and detail_file:
            detail_path = Path(detail_file)
            if not detail_path.is_absolute():
                detail_path = _REPO_ROOT / detail_path
            if detail_path.exists():
                detail = _load_json(detail_path)
                objs = detail.get("objects") or []
                if isinstance(objs, list) and objs:
                    new_objs = [o for o in objs if not (isinstance(o, dict) and o.get("name") == asset_name)]
                    if len(new_objs) != len(objs):
                        removed_any = True
                        detail["objects"] = new_objs
                        if not dry_run:
                            _write_json(detail_path, detail)

        if entry["object_count"] == 0:
            cats_to_delete.append(str(cat_name))

    for cat_name in cats_to_delete:
        cats.pop(cat_name, None)
        removed_any = True

    if removed_any and not dry_run:
        reg["categories"] = cats
        _write_json(reg_path, reg)
    return removed_any


def _delete_old_records(asset_name: str, *, dry_run: bool) -> bool:
    """Delete asset records from both rigid and articulated registries/details."""
    removed_rigid = _remove_asset_from_registry(reg_path=_RIGID_REGISTRY, asset_name=asset_name, dry_run=dry_run)
    removed_art = _remove_asset_from_registry(reg_path=_ART_REGISTRY, asset_name=asset_name, dry_run=dry_run)
    return removed_rigid or removed_art


def _detail_path_for(category: str, obj_type: ObjectType) -> Path:
    if obj_type == "rigid":
        return _TASKGEN_DIR / "objects" / "rigid" / f"rigid_{category}.json"
    return _TASKGEN_DIR / "objects" / "articulated" / f"articulated_{category}.json"


def _ensure_detail_file(detail_path: Path, *, category: dict[str, Any]) -> dict[str, Any]:
    if detail_path.exists():
        detail = _load_json(detail_path)
        if not isinstance(detail, dict):
            raise ValueError(f"Invalid detail file (not a JSON object): {detail_path}")
        if detail.get("category_name") and str(detail.get("category_name")) != str(category["name"]):
            raise ValueError(f"Detail file category_name mismatch: {detail_path}")
        if "objects" not in detail or not isinstance(detail.get("objects"), list):
            detail["objects"] = []
        # Keep existing descriptions if present, otherwise fill from GPT.
        if not isinstance(detail.get("category_description"), str) or not str(detail.get("category_description")).strip():
            detail["category_description"] = category["description"]
        if not isinstance(detail.get("$schema"), str):
            detail["$schema"] = "RoboVerse Category Detail v2.0"
        if not isinstance(detail.get("version"), str):
            detail["version"] = "2.0.0"
        detail["category_name"] = category["name"]
        return detail

    detail = {
        "$schema": "RoboVerse Category Detail v2.0",
        "category_name": category["name"],
        "category_description": category["description"],
        "version": "2.0.0",
        "objects": [],
    }
    return detail


def _build_object_entry(
    *,
    asset_name: str,
    asset_desc: str,
    meta: dict[str, Any],
    mjcf_path: str,
    usd_path: str | None,
    urdf_path: str | None,
    suggested_init: dict[str, Any] | None,
    tabletop: dict[str, Any] | None,
) -> dict[str, Any]:
    # init_state defaults
    z = 0.0
    rot = [1.0, 0.0, 0.0, 0.0]
    joint_states: dict[str, float] | None = None

    if isinstance(suggested_init, dict):
        if isinstance(suggested_init.get("z"), (int, float)):
            z = float(suggested_init["z"])
        r = suggested_init.get("rot")
        if isinstance(r, list) and len(r) == 4 and all(isinstance(v, (int, float)) for v in r):
            rot = [float(v) for v in r]
        js = suggested_init.get("joint_states")
        if isinstance(js, dict) and all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in js.items()):
            joint_states = {str(k): float(v) for k, v in js.items()}

    paths: dict[str, Any] = {"mjcf_path": mjcf_path}
    if usd_path:
        paths["usd_path"] = usd_path
    if urdf_path:
        paths["urdf_path"] = urdf_path

    init_state: dict[str, Any] = {"pos": ["@x", "@y", z], "rot": rot}
    if joint_states:
        init_state["joint_states"] = joint_states

    # Metadata normalization
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    tags_norm = []
    for t in tags:
        if isinstance(t, str) and t.strip():
            tags_norm.append(t.strip())
    meta_out = {
        "tags": tags_norm,
        "color": str(meta.get("color") or "unknown"),
        "material": str(meta.get("material") or "unknown"),
        "size_category": str(meta.get("size_category") or "medium"),
    }

    return {
        "name": asset_name,
        "description": asset_desc,
        "paths": paths,
        "init_state": init_state,
        "scale": [1.0, 1.0, 1.0],
        "metadata": meta_out,
        **({"tabletop": tabletop} if tabletop else {}),
    }


def _upsert_registry_and_detail(
    *,
    obj_type: ObjectType,
    asset_name: str,
    category: dict[str, Any],
    obj: dict[str, Any],
    usd_path: str | None,
    mjcf_path: str,
    urdf_path: str | None,
    dry_run: bool,
) -> tuple[Path, Path]:
    reg_path = _RIGID_REGISTRY if obj_type == "rigid" else _ART_REGISTRY
    reg = _load_registry(reg_path)

    cat_name = str(category["name"])
    cats = reg["categories"]
    if cat_name not in cats:
        detail_path = _detail_path_for(cat_name, obj_type)
        cats[cat_name] = {
            "category_description": str(category["description"]),
            "common_properties": {
                "typical_location": str(category["common_properties"]["typical_location"]),
                "manipulation_type": str(category["common_properties"]["manipulation_type"]),
            },
            "object_names": [asset_name],
            "object_count": 1,
            "detail_file": _to_repo_rel(detail_path),
        }
    else:
        entry = cats[cat_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Registry category entry not an object: {cat_name} in {reg_path}")
        names = entry.get("object_names")
        if not isinstance(names, list):
            names = []
        if asset_name not in names:
            names.append(asset_name)
        # Keep stable-ish ordering.
        names = sorted({str(n) for n in names})
        entry["object_names"] = names
        entry["object_count"] = len(names)
        # Fill missing fields if needed.
        if not isinstance(entry.get("category_description"), str) or not str(entry.get("category_description")).strip():
            entry["category_description"] = str(category["description"])
        if not isinstance(entry.get("common_properties"), dict):
            entry["common_properties"] = {
                "typical_location": str(category["common_properties"]["typical_location"]),
                "manipulation_type": str(category["common_properties"]["manipulation_type"]),
            }
        if not isinstance(entry.get("detail_file"), str) or not str(entry.get("detail_file")).strip():
            entry["detail_file"] = _to_repo_rel(_detail_path_for(cat_name, obj_type))

    detail_file = Path(str(reg["categories"][cat_name]["detail_file"]))
    detail_path = detail_file if detail_file.is_absolute() else (_REPO_ROOT / detail_file)

    detail = _ensure_detail_file(detail_path, category=category)
    objects = detail.get("objects") or []
    if not isinstance(objects, list):
        raise ValueError(f"Invalid detail objects list: {detail_path}")
    if any(isinstance(o, dict) and o.get("name") == asset_name for o in objects):
        raise ValueError(f"Asset already exists in detail file: {asset_name} in {detail_path}")

    entry = _build_object_entry(
        asset_name=asset_name,
        asset_desc=str(obj["description"]).strip(),
        meta=obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
        mjcf_path=mjcf_path,
        usd_path=usd_path,
        urdf_path=urdf_path,
        suggested_init=obj.get("suggested_init_state") if isinstance(obj.get("suggested_init_state"), dict) else None,
        tabletop=obj.get("tabletop") if isinstance(obj.get("tabletop"), dict) else None,
    )
    objects.append(entry)
    detail["objects"] = objects

    if dry_run:
        return reg_path, detail_path

    _write_json(reg_path, reg)
    _write_json(detail_path, detail)
    return reg_path, detail_path


def main() -> None:
    args = tyro.cli(Args)
    asset_root = (_REPO_ROOT / args.asset_root).resolve() if not Path(args.asset_root).is_absolute() else Path(args.asset_root).resolve()
    if not asset_root.exists() or not asset_root.is_dir():
        raise FileNotFoundError(f"asset_root does not exist or is not a directory: {asset_root}")

    if not _RIGID_REGISTRY.exists():
        raise FileNotFoundError(f"Missing rigid registry: {_RIGID_REGISTRY}")
    _ensure_articulated_registry()

    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = openai.OpenAI(api_key=api_key, base_url=args.base_url)

    rigid_reg = _load_registry(_RIGID_REGISTRY)
    art_reg = _load_registry(_ART_REGISTRY)
    cats_summary = _existing_categories_summary(rigid_reg, art_reg)

    system_prompt = f"""\
You are a RoboVerse asset librarian.

Goal: choose a category and write a short description for each asset (by name).
- Prefer an existing category if it fits.
- Otherwise propose a NEW category with a short name in lower_snake_case.
 - IMPORTANT: If the asset is a table / desk / counter used as a tabletop surface, choose category.name="table".

Existing categories:
{cats_summary}

Output MUST be a single JSON object, with NO markdown and NO extra text, exactly in this schema:
{{
  "category": {{
    "name": "lower_snake_case",
    "description": "one sentence",
    "common_properties": {{
      "typical_location": "e.g., kitchen, office, workshop",
      "manipulation_type": "e.g., pick_and_place, pour, open_close, push_pull, turn_knob"
    }}
  }},
  "object": {{
    "description": "one short natural language description of the asset appearance",
    "metadata": {{
      "tags": ["tag1", "tag2"],
      "color": "single color word or 'unknown'",
      "material": "single material word or 'unknown'",
      "size_category": "small|medium|large"
    }},
    "tabletop": {{
      "height": 0.75,
      "range": {{"xmin": -0.4, "xmax": 0.4, "ymin": -0.3, "ymax": 0.3}}
    }},
    "suggested_init_state": {{
      "z": 0.0,
      "rot": [1.0, 0.0, 0.0, 0.0],
      "joint_states": {{}}
    }}
  }}
}}

Notes:
- If you don't know joint names, set joint_states to {{}}.
- Keep category.name concise (<= 24 chars).
- For category.name="table", you may omit object.tabletop; the script will try to auto-estimate from MJCF.
"""

    format_errors: list[str] = []
    skipped_existing: list[str] = []
    processed: list[str] = []
    wrote: list[tuple[str, Path, Path]] = []

    asset_dirs = [p for p in asset_root.iterdir() if p.is_dir()]
    asset_dirs.sort(key=lambda p: p.name.lower())
    if args.only:
        asset_dirs = [p for p in asset_dirs if p.name == args.only]
        if not asset_dirs:
            raise ValueError(f"--only={args.only!r} not found under {asset_root}")
    if args.limit is not None:
        asset_dirs = asset_dirs[: max(0, int(args.limit))]

    for asset_dir in asset_dirs:
        asset_name = asset_dir.name
        mjcf_dir = asset_dir / "mjcf"
        if not mjcf_dir.is_dir():
            format_errors.append(asset_name)
            continue

        mjcf_file = _pick_first_file(mjcf_dir, exts=(".xml",), prefer_stem=asset_name)
        if mjcf_file is None:
            format_errors.append(asset_name)
            continue

        usd_file: Path | None = None
        usd_dir = asset_dir / "usd"
        if usd_dir.is_dir():
            usd_file = _pick_first_file(usd_dir, exts=(".usd", ".usda", ".usdc"), prefer_stem=asset_name)

        urdf_file = _pick_first_file(asset_dir / "urdf", exts=(".urdf",), prefer_stem=asset_name)

        user_prompt = (
            f"asset_name: {asset_name}\n"
            f"asset_root: {_to_repo_rel(asset_dir)}\n"
            f"mjcf_file: {_to_repo_rel(mjcf_file)}\n"
        )
        if usd_file is not None:
            user_prompt += f"usd_file: {_to_repo_rel(usd_file)}\n"
        if urdf_file is not None:
            user_prompt += f"urdf_file: {_to_repo_rel(urdf_file)}\n"

        existing = _find_existing_asset(asset_name, [_RIGID_REGISTRY, _ART_REGISTRY])
        if existing is not None:
            if args.overwrite == "no":
                skipped_existing.append(asset_name)
                continue
            if args.overwrite == "ask":
                reg_p, cat_n = existing
                ok = _ask_yes_no(
                    f"Asset '{asset_name}' already exists (registry={_to_repo_rel(reg_p)}, category={cat_n}). Overwrite?",
                    default=False,
                )
                if not ok:
                    skipped_existing.append(asset_name)
                    continue
            deleted = _delete_old_records(asset_name, dry_run=args.dry_run)
            if not deleted:
                log.warning("Overwrite requested but failed to delete old records for {}", asset_name)

        log.info("Classifying asset: {} (object_type={})", asset_name, args.object_type)
        payload = _call_gpt(
            client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system=system_prompt,
            user=user_prompt,
        )
        category, obj = _validate_gpt_payload(payload, asset_name=asset_name)

        # Table category: ensure tabletop metadata is present (height + range) for downstream tabletop placement.
        if str(category.get("name") or "").strip().lower() == "table":
            tt = obj.get("tabletop")
            if not isinstance(tt, dict) or "height" not in tt or "range" not in tt:
                est = _estimate_tabletop_from_mjcf(mjcf_file, margin_xy=0.03, top_band=0.02)
                if est:
                    obj["tabletop"] = est
                    log.info("Auto-estimated tabletop for {}: {}", asset_name, est)
                else:
                    log.warning("Table asset {} missing tabletop metadata and auto-estimation failed.", asset_name)

        reg_path, detail_path = _upsert_registry_and_detail(
            obj_type=args.object_type,
            asset_name=asset_name,
            category=category,
            obj=obj,
            usd_path=_to_repo_rel(usd_file) if usd_file is not None else None,
            mjcf_path=_to_repo_rel(mjcf_file),
            urdf_path=_to_repo_rel(urdf_file) if urdf_file is not None else None,
            dry_run=args.dry_run,
        )

        wrote.append((asset_name, reg_path, detail_path))
        processed.append(asset_name)

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print("\n=== manage_asset summary ===")
    print(f"asset_root: {asset_root}")
    print(f"processed: {len(processed)}")
    print(f"skipped_existing: {len(skipped_existing)}")
    print(f"format_errors: {len(format_errors)}")
    if args.dry_run:
        print("dry_run: true (no files written)")

    if wrote:
        print("\nWritten/Planned updates:")
        for asset_name, reg_path, detail_path in wrote:
            print(f"- {asset_name}: registry={_to_repo_rel(reg_path)} detail={_to_repo_rel(detail_path)}")

    if skipped_existing:
        print("\nSkipped (already in taskgen_json):")
        for n in skipped_existing:
            print(f"- {n}")

    if format_errors:
        print("\nFormat error assets (missing mjcf/ folder or .xml file):")
        for n in format_errors:
            print(f"- {n}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
