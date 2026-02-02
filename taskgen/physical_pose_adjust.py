"""Manually adjust object initial pose (z + rotation) and scale in taskgen JSON via MuJoCo.

This script:
1) Indexes all assets referenced by taskgen registries (rigid + articulated).
2) Lets you pick an asset from a CLI list (marks assets with `human_check: true`).
3) Opens the asset in a simple MuJoCo scene (x=y fixed at 0,0).
4) Lets you tweak z/rotation and toggle physics.
5) Saves the adjusted quaternion + z + scale back into the corresponding detail JSON.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import pygame
import tyro
from loguru import logger as log

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AssetRef:
    name: str
    category: str
    registry_type: str
    detail_file: Path
    object_index: int


@dataclass
class Args:
    registries: tuple[str, ...] = (
        "taskgen_json/category_registry_objects.json",
        "taskgen_json/category_registry_articulated_objects.json",
    )
    query: str | None = None
    asset: str | None = None  # If set, skip list selection (exact match).
    only_with_mjcf: bool = True
    unchecked_only: bool = False

    step_z: float = 0.005
    step_rot_deg: float = 5.0
    step_scale: float = 0.05
    real_time_hz: float = 60.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _iter_assets_from_registries(registry_paths: list[Path]) -> tuple[list[AssetRef], dict[Path, dict[str, Any]]]:
    """Return (asset_refs, loaded_detail_files_cache)."""
    assets: list[AssetRef] = []
    detail_cache: dict[Path, dict[str, Any]] = {}

    for reg_path in registry_paths:
        reg = _load_json(reg_path)
        registry_type = str(reg.get("registry_type") or "")
        categories = reg.get("categories") or {}
        if not isinstance(categories, dict):
            continue

        for category, entry in categories.items():
            if not isinstance(entry, dict):
                continue
            detail_file = entry.get("detail_file")
            if not detail_file:
                continue
            detail_path = Path(detail_file)
            if not detail_path.exists():
                log.warning("Missing detail_file: {}", detail_path)
                continue
            detail = detail_cache.get(detail_path)
            if detail is None:
                detail = _load_json(detail_path)
                detail_cache[detail_path] = detail

            objs = detail.get("objects") or []
            if not isinstance(objs, list):
                continue
            for i, obj in enumerate(objs):
                if not isinstance(obj, dict):
                    continue
                name = obj.get("name")
                if not isinstance(name, str) or not name:
                    continue
                assets.append(
                    AssetRef(
                        name=name,
                        category=str(category),
                        registry_type=registry_type,
                        detail_file=detail_path,
                        object_index=int(i),
                    )
                )

    assets.sort(key=lambda a: (a.registry_type, a.category, a.name))
    return assets, detail_cache


def _get_obj(detail: dict[str, Any], ref: AssetRef) -> dict[str, Any]:
    objs = detail.get("objects") or []
    if not isinstance(objs, list) or not (0 <= ref.object_index < len(objs)):
        raise ValueError(f"Invalid object_index for {ref.name} in {ref.detail_file}")
    obj = objs[ref.object_index]
    if not isinstance(obj, dict) or obj.get("name") != ref.name:
        # Fallback: locate by name (more robust if ordering changes).
        for o in objs:
            if isinstance(o, dict) and o.get("name") == ref.name:
                return o
        raise ValueError(f"Object {ref.name} not found in {ref.detail_file}")
    return obj


def _is_checked(obj: dict[str, Any]) -> bool:
    return bool(obj.get("human_check") is True)


def _mjcf_path_from_obj(obj: dict[str, Any]) -> Path | None:
    paths = obj.get("paths") or {}
    if not isinstance(paths, dict):
        return None
    mjcf_path = paths.get("mjcf_path")
    if not isinstance(mjcf_path, str) or not mjcf_path:
        return None
    return Path(mjcf_path)


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_norm(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(x) / n for x in q]


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> list[float]:
    ax, ay, az = axis
    s = math.sin(angle_rad / 2.0)
    return _quat_norm([math.cos(angle_rad / 2.0), ax * s, ay * s, az * s])


def _rewrite_mjcf_file_path(base_dir: Path, raw_path: str) -> str:
    """Normalize MJCF `file=` paths.

    - Make relative paths absolute (relative to the current MJCF's directory).
    - If the path is an absolute path from another machine but contains '/roboverse_data/',
      rewrite it to the current repo's roboverse_data path.
    """
    s = (raw_path or "").strip()
    if not s:
        return raw_path

    # If it already exists, keep it.
    try:
        if Path(s).exists():
            return s
    except Exception:
        pass

    # Rewrite common "copied absolute path" patterns to this repo.
    for needle in ("/roboverse_data/", "\\roboverse_data\\"):
        if needle in s:
            tail = s.split(needle, 1)[1]
            candidate = (_REPO_ROOT / "roboverse_data" / Path(tail)).resolve()
            if candidate.exists():
                return str(candidate)

    # Make relative paths absolute.
    if not (s.startswith("/") or (":" in s[:3])):
        return str((base_dir / s).resolve())

    return s


def _ensure_default_geom_class(wrapper: ET.Element, class_name: str, geom_defaults: dict[str, str]) -> None:
    # Find any existing <default class="...">.
    for d in wrapper.iter("default"):
        if (d.get("class") or "") == class_name:
            return

    # Find or create the root <default> block (without a class).
    root_default = None
    for d in list(wrapper):
        if d.tag == "default" and "class" not in d.attrib:
            root_default = d
            break
    if root_default is None:
        root_default = ET.Element("default")
        wrapper.append(root_default)

    cls_default = ET.SubElement(root_default, "default", attrib={"class": class_name})
    ET.SubElement(cls_default, "geom", attrib=geom_defaults)


def _strip_free_joints(elem: ET.Element) -> None:
    """Remove <freejoint/> and joints with type='free' inside elem (in-place)."""
    for parent in elem.iter():
        to_remove: list[ET.Element] = []
        for child in list(parent):
            if child.tag == "freejoint":
                to_remove.append(child)
            elif child.tag == "joint" and (child.get("type") or "").lower() == "free":
                to_remove.append(child)
        for child in to_remove:
            parent.remove(child)


def _build_wrapper_mjcf(asset_xml_path: Path) -> str:
    """Return a standalone MJCF string with a single free joint root and a ground plane."""
    xml_text = asset_xml_path.read_text(encoding="utf-8")
    src_root = ET.fromstring(xml_text)
    base_dir = asset_xml_path.parent

    # Make file paths absolute so the wrapper can live anywhere.
    for node in src_root.iter():
        if "file" in node.attrib:
            node.set("file", _rewrite_mjcf_file_path(base_dir, node.attrib["file"]))

    wrapper = ET.Element("mujoco", attrib={"model": f"pose_adjust:{asset_xml_path.stem}"})

    # Copy all non-worldbody children (compiler/default/asset/etc.) into wrapper.
    src_worldbody = None
    for child in list(src_root):
        if child.tag == "worldbody":
            src_worldbody = child
            continue
        wrapper.append(child)

    # Ensure we have an <asset> block to put a checker/grid ground material into.
    asset_block = next((c for c in list(wrapper) if c.tag == "asset"), None)
    if asset_block is None:
        asset_block = ET.Element("asset")
        wrapper.append(asset_block)

    # A simple checker texture works well as a "grid" floor so the plane is visible.
    # Keep names unique-ish to avoid clobbering if the asset already defines similarly named items.
    ET.SubElement(
        asset_block,
        "texture",
        attrib={
            "name": "rv_ground_tex",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.15 0.15 0.15",
            "rgb2": "0.90 0.90 0.90",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset_block,
        "material",
        attrib={
            "name": "rv_ground_mat",
            "texture": "rv_ground_tex",
            "texrepeat": "12 12",
            "texuniform": "true",
            "reflectance": "0.0",
        },
    )

    # Some assets are MJCF fragments (<mujocoinclude>) and rely on default classes
    # (e.g. class="visual"/"collision") defined elsewhere. Add minimal defaults so they compile.
    _ensure_default_geom_class(
        wrapper,
        "visual",
        {
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "rgba": "0.7 0.7 0.7 1",
        },
    )
    _ensure_default_geom_class(
        wrapper,
        "collision",
        {
            "contype": "1",
            "conaffinity": "1",
            "group": "0",
            "rgba": "0.8 0.8 0.8 0.25",
        },
    )

    worldbody = ET.SubElement(wrapper, "worldbody")
    ET.SubElement(worldbody, "light", attrib={"pos": "0 0 1.6", "dir": "0 0 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        attrib={
            "name": "ground",
            "type": "plane",
            "size": "2 2 0.1",
            "material": "rv_ground_mat",
            "rgba": "1 1 1 1",
        },
    )

    root_body = ET.SubElement(worldbody, "body", attrib={"name": "adjust_root", "pos": "0 0 0", "quat": "1 0 0 0"})
    # Ensure the free body always has a valid inertial, even for purely-visual assets.
    ET.SubElement(root_body, "inertial", attrib={"pos": "0 0 0", "mass": "0.001", "diaginertia": "1e-6 1e-6 1e-6"})
    ET.SubElement(root_body, "freejoint", attrib={"name": "adjust_free"})

    if src_worldbody is not None:
        # Remove any existing free joints; we will control pose via adjust_free.
        _strip_free_joints(src_worldbody)
        for child in list(src_worldbody):
            root_body.append(child)

    return ET.tostring(wrapper, encoding="unicode")


def _find_adjust_free_qposadr(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name == "adjust_free":
                return int(model.jnt_qposadr[j])
    # Fallback: first free joint.
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[j])
    raise ValueError("No free joint found; cannot adjust pose.")


def _find_adjust_free_qveladr(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name == "adjust_free":
                return int(model.jnt_dofadr[j])
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_dofadr[j])
    raise ValueError("No free joint found; cannot adjust pose.")


class PoseAdjustKeyboardClient:
    def __init__(self, asset_name: str, width: int = 520, height: int = 520, title: str = "Pose Adjust Control"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)

        self.asset_name = asset_name
        self.keydown_keys: list[int] = []
        self.quit_requested = False
        self.save_requested = False

        self.instructions = [
            "=== Pose Adjust (z + rot + scale) ===",
            "",
            "Z:",
            "  UP/DOWN     : z +/-",
            "",
            "Rotation:",
            "  LEFT/RIGHT  : yaw (Z) +/-",
            "  U / O       : roll (X) +/-",
            "  I / K       : pitch (Y) +/-",
            "",
            "Scale:",
            "  ] or =      : scale up",
            "  [ or -      : scale down",
            "",
            "Control:",
            "  P           : toggle physics",
            "  L           : toggle XY lock",
            "  SPACE       : settle (1s physics)",
            "  R           : reset to JSON init",
            "  S           : save (write back JSON)",
            "  ESC         : quit (no save)",
        ]

    def update(self) -> None:
        self.keydown_keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            if event.type == pygame.KEYDOWN:
                self.keydown_keys.append(event.key)
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                if event.key == pygame.K_s:
                    self.save_requested = True

    def render(
        self,
        z: float,
        quat: list[float],
        scale: list[float],
        physics_on: bool,
        lock_xy: bool,
        step_z: float,
        step_rot_deg: float,
        step_scale: float,
    ) -> None:
        self.screen.fill((18, 18, 22))

        y = 14
        header = self.font.render(f"Asset: {self.asset_name}", True, (240, 240, 240))
        self.screen.blit(header, (14, y))
        y += 30

        status_lines = [
            f"physics_on: {physics_on}",
            f"lock_xy: {lock_xy}",
            f"z: {z:.4f}   step_z: {step_z:.4f}",
            f"quat [w x y z]: [{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}]",
            f"step_rot_deg: {step_rot_deg:.1f}",
            f"scale [x y z]: [{scale[0]:.3f} {scale[1]:.3f} {scale[2]:.3f}]   step_scale: {step_scale:.3f}",
        ]
        for line in status_lines:
            surf = self.small_font.render(line, True, (210, 210, 210))
            self.screen.blit(surf, (14, y))
            y += 22

        y += 10
        for line in self.instructions:
            color = (200, 200, 200)
            if line.startswith("==="):
                color = (255, 220, 180)
            surf = self.small_font.render(line, True, color)
            self.screen.blit(surf, (14, y))
            y += 22

        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        pygame.quit()


def _pose_scale_from_json(obj: dict[str, Any]) -> tuple[float, list[float], list[float]]:
    init_state = obj.get("init_state") or {}
    if not isinstance(init_state, dict):
        init_state = {}
    pos = init_state.get("pos") or ["@x", "@y", 0.0]
    rot = init_state.get("rot") or [1.0, 0.0, 0.0, 0.0]
    z = 0.0
    if isinstance(pos, list) and len(pos) >= 3 and isinstance(pos[2], (int, float)):
        z = float(pos[2])
    quat = [1.0, 0.0, 0.0, 0.0]
    if isinstance(rot, list) and len(rot) == 4 and all(isinstance(v, (int, float)) for v in rot):
        quat = [float(v) for v in rot]
    scale = obj.get("scale")
    if isinstance(scale, list) and len(scale) == 3 and all(isinstance(v, (int, float)) for v in scale):
        scale_vec = [float(v) for v in scale]
    else:
        scale_vec = [1.0, 1.0, 1.0]
    return z, _quat_norm(quat), scale_vec


def _write_pose_scale_to_json(detail_path: Path, asset_name: str, z: float, quat: list[float], scale: list[float]) -> None:
    detail = _load_json(detail_path)
    objs = detail.get("objects") or []
    if not isinstance(objs, list):
        raise ValueError(f"Invalid objects list in {detail_path}")

    dst_obj: dict[str, Any] | None = None
    for obj in objs:
        if isinstance(obj, dict) and obj.get("name") == asset_name:
            dst_obj = obj
            break
    if dst_obj is None:
        raise ValueError(f"Object {asset_name} not found in {detail_path}")

    init_state = dst_obj.get("init_state")
    if not isinstance(init_state, dict):
        init_state = {}
        dst_obj["init_state"] = init_state

    pos = init_state.get("pos")
    if not isinstance(pos, list) or len(pos) < 3:
        pos = ["@x", "@y", 0.0]
    # Keep @x/@y placeholders; update only z.
    pos = [pos[0] if len(pos) > 0 else "@x", pos[1] if len(pos) > 1 else "@y", float(z)]
    init_state["pos"] = pos
    init_state["rot"] = [float(v) for v in _quat_norm(quat)]

    if not (isinstance(scale, list) and len(scale) == 3 and all(isinstance(v, (int, float)) for v in scale)):
        raise ValueError("scale must be a list[3] of numbers")
    dst_obj["scale"] = [float(scale[0]), float(scale[1]), float(scale[2])]

    dst_obj["human_check"] = True
    _write_json(detail_path, detail)


def _select_asset_interactive(asset_refs: list[AssetRef], detail_cache: dict[Path, dict[str, Any]], args: Args) -> AssetRef | None:
    filtered: list[AssetRef] = []
    q = (args.query or "").strip().lower()

    for ref in asset_refs:
        detail = detail_cache.get(ref.detail_file) or _load_json(ref.detail_file)
        detail_cache[ref.detail_file] = detail
        obj = _get_obj(detail, ref)
        mjcf_path = _mjcf_path_from_obj(obj)
        if args.only_with_mjcf and mjcf_path is None:
            continue
        if args.unchecked_only and _is_checked(obj):
            continue
        if q and q not in ref.name.lower() and q not in ref.category.lower():
            continue
        filtered.append(ref)

    if not filtered:
        log.error("No assets match current filters.")
        return None

    print("\n=== Asset List ===")
    for idx, ref in enumerate(filtered):
        obj = _get_obj(detail_cache[ref.detail_file], ref)
        mark = "✓" if _is_checked(obj) else " "
        print(f"[{idx:03d}] [{mark}] {ref.name}  ({ref.registry_type}/{ref.category})  ::  {ref.detail_file}")

    raw = input("\nSelect asset index (or 'q' to quit): ").strip().lower()
    if raw in {"q", "quit", "exit"}:
        return None
    try:
        choice = int(raw)
    except ValueError:
        log.error("Invalid selection: {}", raw)
        return None
    if choice < 0 or choice >= len(filtered):
        log.error("Selection out of range: {}", choice)
        return None
    return filtered[choice]


def _asset_is_eligible(ref: AssetRef, detail_cache: dict[Path, dict[str, Any]], args: Args) -> bool:
    detail = detail_cache.get(ref.detail_file) or _load_json(ref.detail_file)
    detail_cache[ref.detail_file] = detail
    obj = _get_obj(detail, ref)
    if args.only_with_mjcf and _mjcf_path_from_obj(obj) is None:
        return False
    if args.unchecked_only and _is_checked(obj):
        return False
    q = (args.query or "").strip().lower()
    if q and q not in ref.name.lower() and q not in ref.category.lower():
        return False
    return True


def _find_next_unchecked(
    asset_refs: list[AssetRef],
    detail_cache: dict[Path, dict[str, Any]],
    args: Args,
    after: AssetRef,
) -> AssetRef | None:
    # Advance in the existing sorted order; keep the same filters as selection UI,
    # but always require "unchecked" for auto-advance.
    try:
        start_idx = asset_refs.index(after) + 1
    except ValueError:
        start_idx = 0

    for ref in asset_refs[start_idx:]:
        if not _asset_is_eligible(ref, detail_cache, args):
            continue
        detail = detail_cache[ref.detail_file]
        obj = _get_obj(detail, ref)
        if _is_checked(obj):
            continue
        return ref

    return None


def _run_pose_adjust_ui(
    mjcf_asset_path: Path,
    init_z: float,
    init_quat: list[float],
    init_scale: list[float],
    args: Args,
) -> tuple[float, list[float], list[float]] | None:
    wrapper_xml = _build_wrapper_mjcf(mjcf_asset_path)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(wrapper_xml)
        tmp_xml_path = Path(f.name)

    model = mujoco.MjModel.from_xml_path(str(tmp_xml_path))
    data = mujoco.MjData(model)
    qposadr = _find_adjust_free_qposadr(model)
    qveladr = _find_adjust_free_qveladr(model)

    # MuJoCo model geometry/mesh scaling can be adjusted at runtime by editing model arrays.
    # This is a UI helper (for preview); saved scale is written to taskgen JSON and should be
    # applied by downstream loaders as well.
    base_geom_size = model.geom_size.copy()
    base_mesh_scale = model.mesh_scale.copy()

    adjust_root_bodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "adjust_root")
    if adjust_root_bodyid < 0:
        raise ValueError("Missing body 'adjust_root' in wrapper model; cannot apply scale safely.")

    # Mark bodies that are descendants of `adjust_root` (inclusive).
    nbody = int(model.nbody)
    children: list[list[int]] = [[] for _ in range(nbody)]
    for b in range(nbody):
        p = int(model.body_parentid[b])
        # Be defensive: some malformed models could have parent pointers to self or out of range.
        if p < 0 or p >= nbody or p == b:
            continue
        children[p].append(b)

    body_in_asset = [False] * nbody
    stack = [int(adjust_root_bodyid)]
    while stack:
        cur = stack.pop()
        if cur < 0 or cur >= nbody:
            continue
        if body_in_asset[cur]:
            continue
        body_in_asset[cur] = True
        stack.extend(children[cur])

    geom_in_asset: list[int] = []
    mesh_in_asset: set[int] = set()
    for gi in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gi])
        if 0 <= body_id < len(body_in_asset) and body_in_asset[body_id]:
            geom_in_asset.append(gi)
            if int(model.geom_type[gi]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[gi])
                if mid >= 0:
                    mesh_in_asset.add(mid)

    def apply_scale(scale_vec: list[float]) -> None:
        if not (isinstance(scale_vec, list) and len(scale_vec) == 3 and all(isinstance(v, (int, float)) for v in scale_vec)):
            return
        sx, sy, sz = (max(0.01, float(scale_vec[0])), max(0.01, float(scale_vec[1])), max(0.01, float(scale_vec[2])))
        # Only scale the asset (the subtree under adjust_root), not the ground plane.
        model.geom_size[:] = base_geom_size
        for gi in geom_in_asset:
            model.geom_size[gi] = base_geom_size[gi] * [sx, sy, sz]

        model.mesh_scale[:] = base_mesh_scale
        for mid in mesh_in_asset:
            model.mesh_scale[mid] = base_mesh_scale[mid] * [sx, sy, sz]
        mujoco.mj_forward(model, data)

    def set_pose(z: float, quat: list[float]) -> None:
        data.qpos[qposadr + 0] = 0.0  # x fixed at 0
        data.qpos[qposadr + 1] = 0.0  # y fixed at 0
        data.qpos[qposadr + 2] = float(z)  # z adjustable
        q = _quat_norm(quat)
        data.qpos[qposadr + 3 : qposadr + 7] = q  # type: ignore[assignment]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    set_pose(init_z, init_quat)

    step_z = float(args.step_z)
    step_rot = math.radians(float(args.step_rot_deg))
    step_scale = float(args.step_scale)

    physics_on = False
    lock_xy = True
    scale = [float(init_scale[0]), float(init_scale[1]), float(init_scale[2])]
    apply_scale(scale)
    saved: dict[str, Any] = {"ok": False, "z": 0.0, "quat": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}

    def current_pose() -> tuple[float, list[float]]:
        z = float(data.qpos[qposadr + 2])
        quat = [float(v) for v in data.qpos[qposadr + 3 : qposadr + 7]]
        return z, _quat_norm(quat)

    def rotate(axis: tuple[float, float, float], angle: float) -> None:
        z, q = current_pose()
        dq = _quat_from_axis_angle(axis, angle)
        set_pose(z, _quat_mul(dq, q))

    print(
        "\nMuJoCo Pose Adjust\n"
        "Use the Pygame window for controls/instructions.\n"
        f"Asset: {mjcf_asset_path}\n"
        f"Init z={init_z:.4f}, rot={init_quat}\n"
    )

    control = PoseAdjustKeyboardClient(asset_name=mjcf_asset_path.stem)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            dt = 1.0 / max(1.0, float(args.real_time_hz))
            last = time.time()
            settle_seconds = 1.0
            settle_steps = max(1, int(round(settle_seconds * float(args.real_time_hz))))

            while viewer.is_running() and not control.quit_requested:
                control.update()

                if pygame.K_p in control.keydown_keys:
                    physics_on = not physics_on

                if pygame.K_l in control.keydown_keys:
                    lock_xy = not lock_xy

                if pygame.K_r in control.keydown_keys:
                    set_pose(init_z, init_quat)
                    scale = [float(init_scale[0]), float(init_scale[1]), float(init_scale[2])]
                    apply_scale(scale)

                if pygame.K_UP in control.keydown_keys:
                    z, q = current_pose()
                    set_pose(z + step_z, q)
                if pygame.K_DOWN in control.keydown_keys:
                    z, q = current_pose()
                    set_pose(z - step_z, q)

                if pygame.K_LEFT in control.keydown_keys:
                    rotate((0.0, 0.0, 1.0), +step_rot)
                if pygame.K_RIGHT in control.keydown_keys:
                    rotate((0.0, 0.0, 1.0), -step_rot)

                if pygame.K_u in control.keydown_keys:
                    rotate((1.0, 0.0, 0.0), +step_rot)
                if pygame.K_o in control.keydown_keys:
                    rotate((1.0, 0.0, 0.0), -step_rot)

                if pygame.K_i in control.keydown_keys:
                    rotate((0.0, 1.0, 0.0), +step_rot)
                if pygame.K_k in control.keydown_keys:
                    rotate((0.0, 1.0, 0.0), -step_rot)

                if pygame.K_RIGHTBRACKET in control.keydown_keys or pygame.K_EQUALS in control.keydown_keys:
                    factor = 1.0 + step_scale
                    scale = [max(0.01, float(s) * factor) for s in scale]
                    apply_scale(scale)
                if pygame.K_LEFTBRACKET in control.keydown_keys or pygame.K_MINUS in control.keydown_keys:
                    factor = 1.0 + step_scale
                    scale = [max(0.01, float(s) / factor) for s in scale]
                    apply_scale(scale)

                if pygame.K_SPACE in control.keydown_keys:
                    # Run a short burst of physics to let the object settle, then leave physics off.
                    for _ in range(settle_steps):
                        mujoco.mj_step(model, data)
                        if lock_xy:
                            data.qpos[qposadr + 0] = 0.0
                            data.qpos[qposadr + 1] = 0.0
                            data.qvel[qveladr + 0] = 0.0
                            data.qvel[qveladr + 1] = 0.0
                    physics_on = False

                if control.save_requested:
                    z, q = current_pose()
                    saved["ok"] = True
                    saved["z"] = float(z)
                    saved["quat"] = [float(v) for v in q]
                    saved["scale"] = [float(scale[0]), float(scale[1]), float(scale[2])]
                    viewer.close()
                    break

                now = time.time()
                if now - last < dt:
                    time.sleep(max(0.0, dt - (now - last)))
                    continue
                last = time.time()

                if physics_on:
                    mujoco.mj_step(model, data)
                    if lock_xy:
                        data.qpos[qposadr + 0] = 0.0
                        data.qpos[qposadr + 1] = 0.0
                        data.qvel[qveladr + 0] = 0.0
                        data.qvel[qveladr + 1] = 0.0
                else:
                    mujoco.mj_forward(model, data)

                z, q = current_pose()
                control.render(
                    z=z,
                    quat=q,
                    scale=scale,
                    physics_on=physics_on,
                    lock_xy=lock_xy,
                    step_z=step_z,
                    step_rot_deg=float(args.step_rot_deg),
                    step_scale=step_scale,
                )
                viewer.sync()
    finally:
        control.close()

    if not saved["ok"]:
        return None
    return float(saved["z"]), [float(v) for v in saved["quat"]], [float(v) for v in saved["scale"]]


def main() -> None:
    args = tyro.cli(Args)
    registry_paths = [Path(p) for p in args.registries]
    for p in registry_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    asset_refs, detail_cache = _iter_assets_from_registries(registry_paths)

    ref: AssetRef | None
    if args.asset:
        wanted = args.asset.strip()
        ref = next((a for a in asset_refs if a.name == wanted), None)
        if ref is None:
            raise ValueError(f"Asset not found: {wanted}")
    else:
        ref = _select_asset_interactive(asset_refs, detail_cache, args)
        if ref is None:
            return

    while ref is not None:
        detail = detail_cache.get(ref.detail_file) or _load_json(ref.detail_file)
        detail_cache[ref.detail_file] = detail
        obj = _get_obj(detail, ref)

        mjcf_path = _mjcf_path_from_obj(obj)
        if mjcf_path is None:
            raise ValueError(f"Selected asset has no mjcf_path: {ref.name}")
        if not mjcf_path.exists():
            raise FileNotFoundError(mjcf_path)

        init_z, init_quat, init_scale = _pose_scale_from_json(obj)
        result = _run_pose_adjust_ui(mjcf_path, init_z, init_quat, init_scale, args)
        if result is None:
            log.info("Exited without saving; no JSON updated.")
            return

        z, quat, scale = result
        _write_pose_scale_to_json(ref.detail_file, ref.name, z, quat, scale)
        log.success("Saved pose/scale for {} -> z={}, rot={}, scale={} in {}", ref.name, z, quat, scale, ref.detail_file)

        # Reload the modified file so the next selection sees human_check=true.
        detail_cache[ref.detail_file] = _load_json(ref.detail_file)

        # Auto-advance only when user didn't force a specific asset.
        if args.asset:
            return

        next_ref = _find_next_unchecked(asset_refs, detail_cache, args, after=ref)
        if next_ref is None:
            log.success("No more unchecked assets match current filters.")
            return
        ref = next_ref


if __name__ == "__main__":
    main()
