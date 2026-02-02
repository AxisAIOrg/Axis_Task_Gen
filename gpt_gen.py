#!/usr/bin/env python3
"""GPT-based task generator for RoboVerse.

Three-stage pipeline:
  Stage 1: Category selection based on user prompt
  Stage 2: Task specification (name, instruction, objects, robots)
  Stage 3: Layout generation (positions, rotations)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING

import emoji
import openai
from colorama import Fore, Style

# ======================================
# Difficulty Definitions
# ======================================

DIFFICULTY_DEFINITIONS = {
    1: {
        "name": "Beginner",
        "objects_involved": "1-2",
        "actions": "single action (pick/place)",
        "layout": "objects spread apart",
        "objects_decorative": 0,
    },
    2: {
        "name": "Easy",
        "objects_involved": "2-3",
        "actions": "sequential actions",
        "layout": "medium spacing",
        "objects_decorative": 1,
    },
    3: {
        "name": "Medium",
        "objects_involved": "3-4",
        "actions": "multi-step sequence",
        "layout": "closer spacing",
        "objects_decorative": 2,
    },
    4: {
        "name": "Hard",
        "objects_involved": "4-5",
        "actions": "dependent actions, order matters",
        "layout": "compact layout",
        "objects_decorative": 3,
    },
    5: {
        "name": "Expert",
        "objects_involved": "5+",
        "actions": "long sequence with constraints",
        "layout": "dense layout",
        "objects_decorative": 4,
    },
}

@contextmanager
def _filter_stderr_prefixes(prefixes: tuple[str, ...] = ("INFO", "[INFO]")):
    """Filter noisy native-library INFO lines from stderr (e.g., MuJoCo) without hiding warnings/errors."""
    # IMPORTANT: this is used as a generator-based context manager. It must yield exactly once.
    # Any exception in setup/teardown should fall back to a no-op filter, without yielding again.
    saved_fd = None
    r_fd = None
    t = None

    try:
        saved_fd = os.dup(2)
        r_fd, w_fd = os.pipe()
        os.dup2(w_fd, 2)
        os.close(w_fd)

        def _reader():
            try:
                assert saved_fd is not None
                assert r_fd is not None
                with os.fdopen(r_fd, "rb", closefd=True) as r, os.fdopen(saved_fd, "wb", closefd=False) as w:
                    buf = b""
                    while True:
                        chunk = r.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            text = line.decode(errors="replace")
                            if text.lstrip().startswith(prefixes):
                                continue
                            w.write((text + "\n").encode())
                            w.flush()
                    if buf:
                        text = buf.decode(errors="replace")
                        if not text.lstrip().startswith(prefixes):
                            w.write(text.encode())
                            w.flush()
            except Exception:
                return

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
    except Exception:
        # Setup failed -> run without filtering.
        try:
            yield
        finally:
            return

    try:
        yield
    finally:
        # Best-effort teardown; never raise (raising here breaks generator-based context managers).
        try:
            if saved_fd is not None:
                os.dup2(saved_fd, 2)
        except Exception:
            pass
        try:
            if t is not None:
                t.join(timeout=0.2)
        except Exception:
            pass
        try:
            if saved_fd is not None:
                os.close(saved_fd)
        except Exception:
            pass


def format_difficulty_desc(level: int) -> str:
    """Format difficulty description for prompt injection."""
    d = DIFFICULTY_DEFINITIONS[level]
    return (
        f"Difficulty {level}/5 ({d['name']}):\n"
        f"- Objects decorative: exactly {d['objects_decorative']}"
        f"- Objects involved: {d['objects_involved']}\n"
        f"- Actions: {d['actions']}\n"
        f"- Layout: {d['layout']}\n"
    )

def _ask_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            ans = input(prompt + suffix).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _resolve_tabletop_preference(*, has_table: bool) -> bool:
    """Return whether to enable tabletop mode when a table asset is present.

    Controlled by env var TASKGEN_TABLETOP:
      - "1"/"true"/"yes": always enable
      - "0"/"false"/"no": always disable
      - "ask" or unset: ask if interactive TTY, otherwise disable
    """
    if not has_table:
        return False
    raw = str(os.getenv("TASKGEN_TABLETOP", "")).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    # ask/unset
    if sys.stdin is None or not getattr(sys.stdin, "isatty", lambda: False)():
        return False
    return _ask_yes_no("Detected a table asset in selected objects. Place other objects on the tabletop?", default=False)


# ======================================
# Prompt Templates
# ======================================

STAGE1_SYSTEM_PROMPT = """\
You are a robotic task design expert specializing in object-robot pairing.

## Expertise
- Object affordances and manipulation requirements
- Robot capabilities and workspace constraints

## Task
Analyze user's task and select appropriate categories.

## Table Policy
{table_policy}

{categories_section}

## Hard constraints (MUST follow)
- `object_categories` MUST include at least 1 non-table category that contains task-relevant manipulable objects.
- If you include "table" in `object_categories`, you MUST ALSO include at least one additional non-table category,
  unless the user explicitly asks to manipulate ONLY the table itself.

## Output JSON
{{
  "reasoning": {{
    "task_constraints": "Task requirements analysis",
    "possible_scene": "Envisioned scene and object arrangement",
    "object_needs": "What objects are needed and why"
  }},
  "object_categories": ["cat1", "cat2"],
  "robot_category": "single_arm",
  "task_concept": "One-sentence summary"
}}"""

STAGE2_SYSTEM_PROMPT = """\
You are a robotic task specification expert who creates precise instructions.

## Expertise
- Writing unambiguous manipulation instructions
- Specifying object properties (color, geometry, position)

## {difficulty_desc}

## Context
Task concept: {task_concept}
Available objects: {object_list}
Available robots: {robot_list}

## Table Policy
{table_policy}

## Previous failures (for self-correction; may be empty)
{failure_feedback}

## Output JSON
Return ONLY the JSON object below (no markdown, no code fences, no extra text).
{{
  "reasoning": {{
    "task_design": "Why this task configuration",
    "object_selection": "Why these specific objects",
    "decor_selection": "Why these decorative objects (non-involved)",
    "instruction_rationale": "How instruction guides robot"
  }},
  "task_name": "CamelCaseName",
  "task_language_instruction": "Natural language with object details",
  "robot_involved": [...],
  "objects_involved": [...],
  "decor_objects": [...]
}}

## Hard constraints (MUST follow)
- `objects_involved` are the only task-critical objects (do NOT include decorative objects here).
- `decor_objects` MUST contain exactly {objects_decorative} objects.
- `decor_objects` MUST be a subset of Available objects.
- `decor_objects` MUST NOT overlap with `objects_involved`.
- `robot_involved` MUST be a subset of Available robots.
- You MUST ONLY use exact names from Available objects / Available robots. Do NOT invent names like "decorativeObject1".
- If Available objects are limited, reduce `objects_involved` (but keep task meaningful) so that you can still pick exactly {objects_decorative} decorative objects without inventing names.

## Instruction Examples
1. Pick up the red bottle from the right side, then gently place it into the storage box, \
avoiding collision with the blue cup.
2. Stack the green cube on top of the yellow cube located at the front of the workspace.
3. Grasp the white pitcher from the left, pour water into the glass, then return the pitcher.

## Optional Elements (include as needed)
- Object identifiers: color, shape, size (e.g., 'red bottle', 'small cube')
- Spatial relations: left/right/front/back/near/far
- Action verbs: pick, place, pour, stack, push, slide, rotate
- Motion modifiers: slowly, gently, carefully
- Constraints: avoiding collision, maintaining orientation
- Sequence: then, after, before, finally

You decide which elements to include based on task complexity. \
Not all elements are required - use only what makes the instruction clear and natural."""

STAGE3_SYSTEM_PROMPT = """\
You are a robotic scene layout expert specializing in spatial arrangement.

## Coordinate System (CRITICAL)
Origin: Table center
X-axis: Away from robot (robot's forward, +X = far from robot)
Y-axis: Robot's left (+Y = left side)
Z-axis: Up
Workspace: x,y in [-0.5, 0.5]
Robot position: x ≈ -0.58 (behind workspace)

```
    +Y (robot's left)
         ^
         |
  Robot  |  [Workspace]
   ------|-----> +X (forward)
         |
         v
    -Y (robot's right)
```

## Difficulty: {difficulty}/5
Layout style: {layout_style}
Decorative objects: exactly {objects_decorative}
Expected total objects to place (required + decorative): {num_objects_total}

## Task Context
Task: "{task_name}"
Instruction: "{task_instruction}"
Required objects: {objects_involved}
Decorative objects (MUST include exactly these; do not add/remove/swap): {decor_objects}
Required robots: {robots_involved}

Object library (with fixed z, rot):
{condensed_objs}

Robot library:
{condensed_robots}

## Previous failures (for self-correction; may be empty)
{failure_feedback}

## Output JSON
{{
  "reasoning": {{
    "spatial_analysis": "How to arrange based on instruction",
    "instruction_alignment": "How layout matches spatial descriptions"
  }},
  "workspace": {{
    "xmin": -0.5,
    "xmax": 0.5,
    "ymin": -0.5,
    "ymax": 0.5,
    "surface": "ground",
    "table_body_or_geom_name": "table"
  }},
  "min_clearance": {min_clearance},
  "max_attempts": 200,
  "decor_density": "medium",
  "objects": [
    {{
      "name": "obj1",
      "xy": [x, y],
      "priority": 10,
      "relative_to": null
    }}
  ],
  "robots": [{{"name": "robot1", "pos": [x, y, z], "rot": [w, x, y, z], "dof_pos": {{...}}}}, ...]
}}

## Rules
- If instruction says 'left', use positive Y
- If instruction says 'right', use negative Y
- If instruction says 'front/far', use positive X
- This stage outputs intent + coarse layout only (XY + yaw). Do NOT try to perfectly solve all geometric constraints.
- Prefer semantic correctness (left/right/front/back, near/far, inside/next-to) and a natural non-grid style.
- Use `priority` to indicate which objects must keep their intended relative positions (required objects higher, decorative lower).
- Still try to keep objects reasonably separated: suggested minimum clearance in XY is {min_clearance}m.
- For objects: output `xy` only; z and rot will be taken from the object library downstream
- For robots: copy pos, rot, dof_pos exactly from library
- HARD CONSTRAINT: Output objects MUST be exactly (Required objects ∪ Decorative objects) with no extras or omissions.
- PLACEMENT: Avoid grid-aligned placement. Use random offsets to create natural, cluttered scenes"""

STAGE4_SYSTEM_PROMPT = """\
You are a robotic scene layout fixer.

## Task
One object failed to be inserted into the scene due to a physics validation error.
You MUST propose a new placement for ONLY that object.

## Hard constraints
- Output must be valid JSON (no markdown).
- You may only modify the failing object's (x, y).
- Do NOT add/remove/rename any objects.
- Keep (x,y) strictly inside workspace bounds.
- z will be deterministically recomputed by the program (ignore z).
- Do NOT propose any rotations.

## Context (JSON)
{context_json}

## Output JSON (ONLY)
{{
  "name": "obj_new",
  "pos": [x, y],
  "reason": "one sentence"
}}
"""


def _env_flag(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, "")).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def format_stage1_prompt(obj_cats: dict, robot_cats: dict, *, allow_table: bool) -> str:
    """Render Stage 1 prompt with category data."""
    categories_section = (
        f"Object Categories:\n{json.dumps(obj_cats, indent=2)}\n\n"
        f"Robot Categories:\n{json.dumps(robot_cats, indent=2)}"
    )
    table_policy = (
        "Table objects are allowed and REQUIRED: you MUST include category `table` in `object_categories`.\n"
        "You must also include at least one additional non-table category with manipulable objects (so tasks are possible)."
        if allow_table
        else "Table objects are NOT allowed. Do NOT select category `table` or any table-like category."
    )
    return STAGE1_SYSTEM_PROMPT.format(categories_section=categories_section, table_policy=table_policy)


def format_stage2_prompt(
    object_list: list,
    robot_list: list,
    task_concept: str,
    difficulty: int,
    *,
    allow_table: bool,
    table_object_names: list[str],
    failure_feedback: str = "",
) -> str:
    """Render Stage 2 prompt with task context."""
    if allow_table:
        table_list = ", ".join(table_object_names[:40]) + ("..." if len(table_object_names) > 40 else "")
        table_policy = (
            "You MUST include exactly ONE tabletop support object in `objects_involved`.\n"
            "That tabletop object MUST be chosen from the Table objects list below (exact name).\n"
            "Do NOT put any table object into `decor_objects`.\n"
            f"Table objects: [{table_list}]"
        )
    else:
        table_policy = "You MUST NOT select any table object."
    return STAGE2_SYSTEM_PROMPT.format(
        difficulty_desc=format_difficulty_desc(difficulty),
        objects_decorative=DIFFICULTY_DEFINITIONS[difficulty]["objects_decorative"],
        task_concept=task_concept,
        object_list=object_list,
        robot_list=robot_list,
        table_policy=table_policy,
        failure_feedback=failure_feedback.strip() or "(none)",
    )


def format_stage3_prompt(
    partial: dict, condensed_objs: dict, condensed_robots: dict, difficulty: int, failure_feedback: str = ""
) -> str:
    """Render Stage 3 prompt with layout context."""
    objects_decorative = DIFFICULTY_DEFINITIONS[difficulty]["objects_decorative"]
    num_required = len(partial.get("objects_involved") or [])
    num_objects_total = num_required + int(objects_decorative)

    # Heuristic: suggest more clearance as object count grows (validator/repairer will enforce).
    if num_objects_total <= 4:
        min_clearance = 0.10
    elif num_objects_total <= 7:
        min_clearance = 0.12
    elif num_objects_total <= 10:
        min_clearance = 0.14
    else:
        min_clearance = 0.16

    return STAGE3_SYSTEM_PROMPT.format(
        difficulty=difficulty,
        layout_style=DIFFICULTY_DEFINITIONS[difficulty]["layout"],
        objects_decorative=objects_decorative,
        num_objects_total=num_objects_total,
        min_clearance=f"{min_clearance:.2f}",
        task_name=partial["task_name"],
        task_instruction=partial["task_language_instruction"],
        objects_involved=partial["objects_involved"],
        decor_objects=partial.get("decor_objects", []),
        robots_involved=partial["robot_involved"],
        condensed_objs=json.dumps(condensed_objs, indent=2),
        condensed_robots=json.dumps(condensed_robots, indent=2),
        failure_feedback=failure_feedback.strip() or "(none)",
    )


def format_stage4_prompt(context: dict) -> str:
    return STAGE4_SYSTEM_PROMPT.format(context_json=json.dumps(context, indent=2))


# ======================================
# Layout Validation / Repair
# ======================================


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    """Quaternion multiply (w,x,y,z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]

def _quat_norm(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / n for v in q]


def _yaw_quat(yaw: float) -> list[float]:
    """Yaw-only quaternion about +Z in (w,x,y,z)."""
    half = 0.5 * float(yaw)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _spiral_offsets(step: float = 0.03, turns: int = 10, points_per_turn: int = 18) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    for t in range(1, turns + 1):
        r = step * t
        for k in range(points_per_turn):
            ang = 2.0 * math.pi * (k / points_per_turn) + (t * 0.37)
            offsets.append((r * math.cos(ang), r * math.sin(ang)))
    return offsets


class LayoutValidator:
    """Validate a concrete layout with either MuJoCo (preferred) or a fast 2D fallback."""

    # Only treat as "collision" when contact penetration is deeper than this (meters).
    COLLISION_DIST: float = -2e-2
    # In insertion, Z is treated as "support_z + delta_z".
    Z_START_ABOVE_SUPPORT: float = 0.03
    Z_STEP_ABOVE_SUPPORT: float = 0.01
    # Hard cap on delta_z used during insertion (meters).
    MAX_Z_ABOVE_SUPPORT: float = 1.5

    def __init__(self, use_mujoco: bool = True):
        self._use_mujoco = use_mujoco
        self._mujoco_ready = False
        self._handler = None
        self._owner_prefix: dict[str, str] = {}
        self._robot_prefix: dict[str, str] = {}
        # Cache key for the currently loaded MuJoCo model. Must include any metadata that affects geometry,
        # otherwise we can silently reuse a model with stale scale.
        self._loaded_object_signature: frozenset[tuple] | None = None

    def validate_fast(
        self,
        *,
        objects: list[dict],
        workspace: dict,
        min_clearance: float,
    ) -> dict:
        result = self._validate_2d(objects=objects, workspace=workspace, min_clearance=min_clearance)
        result["mode"] = "2d"
        return result

    def _validate_2d(self, *, objects: list[dict], workspace: dict, min_clearance: float) -> dict:
        errors: list[dict] = []
        xmin, xmax = float(workspace["xmin"]), float(workspace["xmax"])
        ymin, ymax = float(workspace["ymin"]), float(workspace["ymax"])

        for o in objects:
            x, y, _ = o["pos"]
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                errors.append({"type": "out_of_bounds", "obj": o["name"]})

        for i in range(len(objects)):
            xi, yi, _ = objects[i]["pos"]
            for j in range(i + 1, len(objects)):
                xj, yj, _ = objects[j]["pos"]
                dx, dy = xi - xj, yi - yj
                if (dx * dx + dy * dy) ** 0.5 < min_clearance:
                    errors.append({"type": "collision", "a": objects[i]["name"], "b": objects[j]["name"]})

        return {"ok": len(errors) == 0, "errors": errors}

    def validate_with_physics(
        self,
        *,
        objects: list[dict],
        robots: list[dict],
        workspace: dict,
        mujoco_assets: dict[str, dict],
        steps: int = 300,
        pen_tol: float = 5e-4,
        deep_pen: float = 3e-3,
        stop_vel: float = 1e-2,
        stop_frames: int = 30,
    ) -> dict:
        """MuJoCo-based validation: t=0 contact check + settle under gravity with continuous monitoring."""
        if not self._use_mujoco:
            return {"ok": False, "errors": [{"type": "validator_disabled"}], "mode": "mujoco_physics"}

        try:
            return self._validate_with_physics(
                objects=objects,
                robots=robots,
                workspace=workspace,
                mujoco_assets=mujoco_assets,
                steps=steps,
                pen_tol=pen_tol,
                deep_pen=deep_pen,
                stop_vel=stop_vel,
                stop_frames=stop_frames,
            )
        except Exception as e:
            return {"ok": False, "errors": [{"type": "validator_exception", "message": str(e)}], "mode": "mujoco_physics"}

    def validate_with_incremental_insertion(
        self,
        *,
        objects: list[dict],
        robots: list[dict],
        workspace: dict,
        mujoco_assets: dict[str, dict],
        involved_objects: list[str],
        decor_objects: list[str],
        debug: bool = False,
        steps: int = 10_000,
        pen_tol: float = 2e-4,
        deep_pen: float = 3e-3,
        ground_pen_ok: float = 2e-2,
        eps_z: float = 2e-3,
        per_obj_attempts: int = 80,
        settle_stop_vel: float = 1e-1,
        settle_stop_frames: int = 60,
    ) -> dict:
        """Incremental insertion + settle validation.

        - Teleports non-inserted objects to an isolation zone.
        - Inserts objects one-by-one and attributes collisions to the newly inserted object.
        - During insertion, z starts at support_z + 0.03 and is lifted by +0.01 only if colliding with support.
        - Set membership is never changed.
        """
        if not self._use_mujoco:
            return {"ok": False, "errors": [{"type": "validator_disabled"}], "mode": "mujoco_incremental"}
        try:
            return self._validate_with_incremental_insertion(
                objects=objects,
                robots=robots,
                workspace=workspace,
                mujoco_assets=mujoco_assets,
                involved_objects=involved_objects,
                decor_objects=decor_objects,
                debug=debug,
                steps=steps,
                pen_tol=pen_tol,
                deep_pen=deep_pen,
                ground_pen_ok=ground_pen_ok,
                eps_z=eps_z,
                per_obj_attempts=per_obj_attempts,
                settle_stop_vel=settle_stop_vel,
                settle_stop_frames=settle_stop_frames,
            )
        except Exception as e:
            return {"ok": False, "errors": [{"type": "validator_exception", "message": str(e)}], "mode": "mujoco_incremental"}

    def estimate_tabletop_workspace(
        self,
        *,
        workspace: dict,
        robots: list[dict],
        mujoco_assets: dict[str, dict],
        margin_xy: float = 0.03,
        top_band: float = 0.02,
    ) -> dict[str, float] | None:
        """Best-effort inference of tabletop bounds and height from the MuJoCo scene.

        Returns:
          {"xmin","xmax","ymin","ymax","tabletop_z"} or None if no table geoms found.
        """
        if not self._use_mujoco:
            return None
        try:
            self._ensure_mujoco_handler(mujoco_assets=mujoco_assets, robots=robots)
        except Exception:
            return None

        handler = self._handler
        if handler is None:
            return None
        model = getattr(handler, "mj_model", None) or getattr(handler, "native_model", None)
        if model is None:
            return None
        try:
            import mujoco  # local import (heavy)
        except Exception:
            return None

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Identify robot geoms by prefix so we don't confuse them as table.
        robot_geom_ids: set[int] = set()
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            for _, prefix in self._robot_prefix.items():
                if gname.startswith(prefix):
                    robot_geom_ids.add(gid)

        table_hint = str(workspace.get("table_body_or_geom_name") or "table").lower()
        table_geom_ids: list[int] = []
        for gid in range(model.ngeom):
            if gid in robot_geom_ids:
                continue
            gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if table_hint and table_hint in gname:
                table_geom_ids.append(gid)
                continue
            if ("table" in gname) or ("counter" in gname) or ("desk" in gname):
                table_geom_ids.append(gid)

        if not table_geom_ids:
            return None

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
            # Mesh/other: approximate with rbound (bounding sphere).
            try:
                return z0 + float(model.geom_rbound[gid])
            except Exception:
                return z0

        top_z = max(_geom_top_z(gid) for gid in table_geom_ids)
        band = max(0.0, float(top_band))
        top_gids = [gid for gid in table_geom_ids if _geom_top_z(gid) >= (top_z - band)]
        if not top_gids:
            top_gids = list(table_geom_ids)

        xmin = float("inf")
        xmax = float("-inf")
        ymin = float("inf")
        ymax = float("-inf")

        for gid in top_gids:
            cx, cy, _ = [float(v) for v in data.geom_xpos[gid]]
            gtype = int(model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue

            # Conservative XY extents per geom.
            ex = ey = None
            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                sx, sy, sz = [float(v) for v in model.geom_size[gid]]
                # geom_xmat is row-major 3x3 rotation.
                R = [float(v) for v in data.geom_xmat[gid]]
                r00, r01, r02, r10, r11, r12 = abs(R[0]), abs(R[1]), abs(R[2]), abs(R[3]), abs(R[4]), abs(R[5])
                # extents = |R| @ halfsizes
                ex = r00 * sx + r01 * sy + r02 * sz
                ey = r10 * sx + r11 * sy + r12 * sz
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                r = float(model.geom_size[gid][0])
                ex = ey = r
            else:
                # Mesh/cyl/capsule/etc: use bounding sphere radius for XY bounds.
                try:
                    r = float(model.geom_rbound[gid])
                except Exception:
                    r = 0.0
                ex = ey = max(0.0, r)

            xmin = min(xmin, cx - float(ex))
            xmax = max(xmax, cx + float(ex))
            ymin = min(ymin, cy - float(ey))
            ymax = max(ymax, cy + float(ey))

        if not (math.isfinite(xmin) and math.isfinite(xmax) and math.isfinite(ymin) and math.isfinite(ymax)):
            return None

        m = max(0.0, float(margin_xy))
        return {
            "xmin": float(xmin + m),
            "xmax": float(xmax - m),
            "ymin": float(ymin + m),
            "ymax": float(ymax - m),
            "tabletop_z": float(top_z),
        }

    @staticmethod
    def estimate_table_asset_surface(
        *,
        mjcf_path: str,
        margin_xy: float = 0.03,
        top_band: float = 0.02,
    ) -> dict[str, float] | None:
        """Estimate a table-like asset's top surface bounds in its own model frame.

        Returns {"xmin","xmax","ymin","ymax","tabletop_z"} in the asset model frame.
        """
        try:
            import mujoco  # local import
        except Exception:
            return None

        p = Path(mjcf_path)
        if not p.exists():
            return None

        try:
            model = mujoco.MjModel.from_xml_path(str(p))
        except Exception:
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

            ex = ey = None
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
                try:
                    r = float(model.geom_rbound[gid])
                except Exception:
                    r = 0.0
                ex = ey = max(0.0, r)

            xmin = min(xmin, cx - float(ex))
            xmax = max(xmax, cx + float(ex))
            ymin = min(ymin, cy - float(ey))
            ymax = max(ymax, cy + float(ey))

        if not (math.isfinite(xmin) and math.isfinite(xmax) and math.isfinite(ymin) and math.isfinite(ymax)):
            return None

        m = max(0.0, float(margin_xy))
        return {
            "xmin": float(xmin + m),
            "xmax": float(xmax - m),
            "ymin": float(ymin + m),
            "ymax": float(ymax - m),
            "tabletop_z": float(top_z),
        }

    def _ensure_mujoco_handler(self, *, mujoco_assets: dict[str, dict], robots: list[dict], scene: str | None = None) -> None:
        def _norm_scale(meta: dict) -> float | tuple[float, float, float]:
            s = meta.get("scale")
            if isinstance(s, (int, float)):
                return float(s)
            if isinstance(s, (list, tuple)) and len(s) == 3 and all(isinstance(v, (int, float)) for v in s):
                return (float(s[0]), float(s[1]), float(s[2]))
            return 1.0

        scene_key = str(scene) if scene is not None else "<none>"
        desired = frozenset(
            (scene_key, str(name), str((meta or {}).get("mjcf_path") or ""), _norm_scale(meta or {})) for name, meta in mujoco_assets.items()
        )
        if self._mujoco_ready and self._loaded_object_signature == desired:
            return
        # Different object set: rebuild handler.
        self._mujoco_ready = False
        self._handler = None
        self._owner_prefix = {}
        self._loaded_object_signature = desired

        try:
            from metasim.scenario.objects import RigidObjCfg
            from metasim.scenario.scenario import ScenarioCfg
            from metasim.utils.setup_util import get_handler
        except Exception as e:
            raise RuntimeError("MuJoCo validator requires metasim + mujoco dependencies.") from e

        robot_names = [r["name"] for r in robots] if robots else ["franka"]
        scenario_objects = []
        for name, meta in mujoco_assets.items():
            mjcf_path = meta.get("mjcf_path")
            if not mjcf_path:
                continue
            scale = meta.get("scale")
            if isinstance(scale, (int, float)):
                scale = float(scale)
            elif isinstance(scale, (list, tuple)) and len(scale) == 3 and all(isinstance(v, (int, float)) for v in scale):
                scale = (float(scale[0]), float(scale[1]), float(scale[2]))
            else:
                scale = 1.0
            scenario_objects.append(
                RigidObjCfg(
                    name=name,
                    mjcf_path=mjcf_path,
                    urdf_path=meta.get("urdf_path"),
                    usd_path=meta.get("usd_path"),
                    scale=scale,
                )
            )

        scenario = ScenarioCfg(
            simulator="mujoco",
            headless=True,
            num_envs=1,
            scene=scene,
            robots=robot_names,
            objects=scenario_objects,
        )
        handler = get_handler(scenario)
        owner_prefix: dict[str, str] = {}
        def _ensure_trailing_slash(s: str) -> str:
            return s if s.endswith("/") else (s + "/")

        try:
            # Prefer attached full identifiers (unique even when MJCF model names collide like "textured_1").
            for cfg, full_id in zip(handler.objects, getattr(handler, "object_body_names", [])):
                if getattr(cfg, "name", None) in mujoco_assets and isinstance(full_id, str):
                    owner_prefix[str(cfg.name)] = _ensure_trailing_slash(full_id)
        except Exception:
            owner_prefix = {}
        # Fallback to model names if needed.
        for name in mujoco_assets.keys():
            if name in owner_prefix:
                continue
            try:
                owner_prefix[name] = _ensure_trailing_slash(handler.mj_objects[name].model)
            except Exception:
                pass

        robot_prefix: dict[str, str] = {}
        try:
            mujoco_robot_names = getattr(handler, "_mujoco_robot_names", [])
            for i, cfg in enumerate(handler.robots):
                if i < len(mujoco_robot_names) and isinstance(mujoco_robot_names[i], str):
                    robot_prefix[str(cfg.name)] = _ensure_trailing_slash(mujoco_robot_names[i])
        except Exception:
            robot_prefix = {}

        self._handler = handler
        self._owner_prefix = owner_prefix
        self._robot_prefix = robot_prefix
        self._mujoco_ready = True

    def _owner_from_body_name(self, body_name: str) -> str | None:
        for obj_name, prefix in self._owner_prefix.items():
            if body_name.startswith(prefix) or body_name.startswith(prefix.rstrip("/")):
                return obj_name
        return None

    def _validate_with_physics(
        self,
        *,
        objects: list[dict],
        robots: list[dict],
        workspace: dict,
        mujoco_assets: dict[str, dict],
        steps: int,
        pen_tol: float,
        deep_pen: float,
        stop_vel: float,
        stop_frames: int,
    ) -> dict:
        import mujoco

        with _filter_stderr_prefixes():
            # Match the default "human-check" MuJoCo env: no scene -> handler adds its own default ground plane.
            self._ensure_mujoco_handler(mujoco_assets=mujoco_assets, robots=robots, scene=None)
            handler = self._handler
            assert handler is not None

            # IMPORTANT: use the same dm_control physics model/data that tasks use (human-check/replay),
            # not the separately-exported native model, to avoid pose mismatches.
            phy = getattr(handler, "physics", None)
            if phy is None or getattr(getattr(phy, "model", None), "ptr", None) is None or getattr(getattr(phy, "data", None), "ptr", None) is None:
                raise RuntimeError("MuJoCo dm_control physics not available on handler.")
            model = phy.model.ptr
            data = phy.data.ptr

            # Ensure gravity is enabled for validation (settle relies on it).
            # MuJoCo gravity is a model property, so restore it after validation.
            orig_gravity = tuple(float(x) for x in model.opt.gravity)
            gravity_forced = False
            if abs(orig_gravity[0]) + abs(orig_gravity[1]) + abs(orig_gravity[2]) < 1e-8:
                model.opt.gravity[:] = [0.0, 0.0, -9.81]
                gravity_forced = True

            mujoco.mj_resetData(model, data)
            data.qvel[:] = 0
            if hasattr(data, "ctrl") and data.ctrl is not None and data.ctrl.size > 0:
                data.ctrl[:] = 0

        # Build joint mappings for free joints (objects + robots).
        obj_prefix = {name: self._owner_prefix[name] for name in mujoco_assets.keys() if name in self._owner_prefix}
        robot_prefix = {r["name"]: self._robot_prefix[r["name"]] for r in robots if r.get("name") in self._robot_prefix}

        def _find_free_joint_id(prefix: str) -> int | None:
            p0 = prefix.rstrip("/")
            for jid in range(model.njnt):
                if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                # Prefer matching by the joint's owning body name (more reliable than joint naming).
                try:
                    body_id = int(model.jnt_bodyid[jid])
                    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                except Exception:
                    bname = None
                if bname and (bname.startswith(prefix) or bname.startswith(p0)):
                    return jid

                # Fallback: match by joint name prefix.
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jname and (jname.startswith(prefix) or jname.startswith(p0)):
                    return jid
            return None

        obj_free_jid: dict[str, int] = {}
        obj_qposadr: dict[str, int] = {}
        obj_dofadr: dict[str, int] = {}
        for name, prefix in obj_prefix.items():
            jid = _find_free_joint_id(prefix)
            if jid is None:
                continue
            obj_free_jid[name] = jid
            obj_qposadr[name] = int(model.jnt_qposadr[jid])
            obj_dofadr[name] = int(model.jnt_dofadr[jid])

        # Apply object poses (free joint qpos = [x y z qw qx qy qz]).
        for o in objects:
            name = o.get("name")
            if name not in obj_qposadr:
                continue
            adr = obj_qposadr[name]
            pos = o.get("pos")
            rot = o.get("rot")
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                continue
            if not isinstance(rot, (list, tuple)) or len(rot) != 4:
                continue
            data.qpos[adr : adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
            data.qpos[adr + 3 : adr + 7] = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]

        # Apply robot root pose if free joint exists, and set joint dof_pos if provided.
        for r in robots:
            rname = r.get("name")
            if not isinstance(rname, str) or rname not in robot_prefix:
                continue
            prefix = robot_prefix[rname]
            rjid = _find_free_joint_id(prefix)
            if rjid is not None:
                adr = int(model.jnt_qposadr[rjid])
                pos = r.get("pos")
                rot = r.get("rot")
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    data.qpos[adr : adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
                if isinstance(rot, (list, tuple)) and len(rot) == 4:
                    data.qpos[adr + 3 : adr + 7] = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]

            dof_pos = r.get("dof_pos") or {}
            if isinstance(dof_pos, dict):
                for jn, jv in dof_pos.items():
                    try:
                        full = f"{prefix}{jn}"
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
                        qadr = int(model.jnt_qposadr[jid])
                        # Assume 1-DoF hinge/slide joints here.
                        data.qpos[qadr] = float(jv)
                    except Exception:
                        continue

        mujoco.mj_forward(model, data)

        # Build geom sets for filtering.
        object_geom_ids: set[int] = set()
        for name, prefix in obj_prefix.items():
            for gid in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if gname and gname.startswith(prefix):
                    object_geom_ids.add(gid)

        robot_geom_ids: set[int] = set()
        for rname, prefix in robot_prefix.items():
            for gid in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if gname and gname.startswith(prefix):
                    robot_geom_ids.add(gid)

        table_geom_ids: set[int] = set()
        ground_geom_ids: set[int] = set()
        for gid in range(model.ngeom):
            if gid in object_geom_ids or gid in robot_geom_ids:
                continue
            gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            gtype = int(model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE) or "ground" in gname or "floor" in gname:
                ground_geom_ids.add(gid)
            if "table" in gname or "counter" in gname or "desk" in gname:
                table_geom_ids.add(gid)

        def _geom_kind(gid: int) -> str:
            if gid in object_geom_ids:
                return "object"
            if gid in table_geom_ids:
                return "table"
            if gid in ground_geom_ids:
                return "ground"
            if gid in robot_geom_ids:
                return "robot"
            return "scene"

        def _owner_from_geom(gid: int) -> str | None:
            body_id = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not bname:
                return None
            return self._owner_from_body_name(bname)

        def _check_contacts() -> dict | None:
            # Returns a single error dict if any violation found.
            for i in range(int(data.ncon)):
                con = data.contact[i]
                g1 = int(con.geom1)
                g2 = int(con.geom2)
                dist = float(con.dist)
                k1 = _geom_kind(g1)
                k2 = _geom_kind(g2)

                # Ignore robot-table contacts (not relevant for layout validation).
                if (k1 == "robot" and k2 == "table") or (k1 == "table" and k2 == "robot"):
                    continue

                # Tolerances: allow small penetration into ground (common in MJCF scenes) but keep strict elsewhere.
                is_object_object = (k1 == "object" and k2 == "object")
                is_object_ground = ("object" in (k1, k2) and "ground" in (k1, k2))
                is_object_support = ("object" in (k1, k2)) and ("robot" not in (k1, k2)) and not is_object_object

                min_sep = abs(float(self.COLLISION_DIST))  # treat dist > -min_sep as separated everywhere
                support_pen_tol = 1e-2 if is_object_ground else float(pen_tol)
                support_pen_tol = max(support_pen_tol, min_sep)
                deep_pen_tol = max(float(deep_pen), min_sep)
                if is_object_ground:
                    # Unified threshold: treat dist >= COLLISION_DIST as separated.
                    pass

                # Deep penetration is always suspicious (including allowed table/ground contacts),
                # but uses kind-specific thresholds above.
                if dist < -deep_pen_tol:
                    obj = None
                    if k1 == "object":
                        obj = _owner_from_geom(g1)
                    elif k2 == "object":
                        obj = _owner_from_geom(g2)
                    return {
                        "type": "deep_penetration",
                        "obj": obj,
                        "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                        "dist": dist,
                        "kinds": [k1, k2],
                    }

                # Forbidden: any object-object contact (even just touching).
                if is_object_object and dist < float(self.COLLISION_DIST):
                    a = _owner_from_geom(g1)
                    b = _owner_from_geom(g2)
                    if a and b and a != b:
                        return {"type": "collision", "a": a, "b": b, "dist": dist}

                # Forbidden: object-robot contact (treat as invalid initial/settle state).
                if ("object" in (k1, k2) and "robot" in (k1, k2)) and dist < float(self.COLLISION_DIST):
                    a = _owner_from_geom(g1) if k1 == "object" else _owner_from_geom(g2)
                    b = "robot"
                    if a:
                        return {"type": "collision_with_robot", "obj": a, "dist": dist}

                # Penetration tolerance for object-support contacts:
                # allow small numerical penetration, but fail if the object sinks into support/scene geometry.
                if is_object_support and dist < -support_pen_tol:
                    a = _owner_from_geom(g1) if k1 == "object" else _owner_from_geom(g2)
                    return {
                        "type": "support_penetration",
                        "obj": a,
                        "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                        "dist": dist,
                        "kinds": [k1, k2],
                    }

            return None

        def _check_workspace() -> dict | None:
            xmin, xmax = float(workspace["xmin"]), float(workspace["xmax"])
            ymin, ymax = float(workspace["ymin"]), float(workspace["ymax"])
            for name, adr in obj_qposadr.items():
                x = float(data.qpos[adr + 0])
                y = float(data.qpos[adr + 1])
                z = float(data.qpos[adr + 2])
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    return {"type": "out_of_bounds", "obj": name}
                # Catch gross fall-through (below global ground plane) even if contacts are missing.
                if z < -0.02:
                    return {"type": "below_ground", "obj": name, "z": z}
            return None

        # t=0 checks
        err0 = _check_contacts()
        if err0 is not None:
            if gravity_forced:
                model.opt.gravity[:] = orig_gravity
            if isinstance(err0, dict) and err0.get("type") in {"deep_penetration", "support_penetration"}:
                obj = err0.get("obj")
                if isinstance(obj, str) and obj in obj_prefix and obj not in obj_qposadr:
                    err0 = dict(err0)
                    err0["note"] = "Object pose may be unmapped (no freejoint found); repair may not change this contact."
            return {
                "ok": False,
                "errors": [err0],
                "mode": "mujoco_physics",
                "phase": "t0",
                "settled_objects": {},
                "gravity": tuple(float(x) for x in model.opt.gravity),
                "gravity_forced": gravity_forced,
                "mapped_objects": sorted(list(obj_qposadr.keys())),
                "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
            }
        err0b = _check_workspace()
        if err0b is not None:
            if gravity_forced:
                model.opt.gravity[:] = orig_gravity
            return {
                "ok": False,
                "errors": [err0b],
                "mode": "mujoco_physics",
                "phase": "t0",
                "settled_objects": {},
                "gravity": tuple(float(x) for x in model.opt.gravity),
                "gravity_forced": gravity_forced,
                "mapped_objects": sorted(list(obj_qposadr.keys())),
                "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
            }

        # settle loop
        stable = 0
        steps_run = 0
        for t in range(int(steps)):
            mujoco.mj_step(model, data)
            steps_run = t + 1

            err = _check_contacts()
            if err is not None:
                err["step"] = t + 1
                if gravity_forced:
                    model.opt.gravity[:] = orig_gravity
                if isinstance(err, dict) and err.get("type") in {"deep_penetration", "support_penetration"}:
                    obj = err.get("obj")
                    if isinstance(obj, str) and obj in obj_prefix and obj not in obj_qposadr:
                        err = dict(err)
                        err["note"] = "Object pose may be unmapped (no freejoint found); repair may not change this contact."
                return {
                    "ok": False,
                    "errors": [err],
                    "mode": "mujoco_physics",
                    "phase": "settle",
                    "settled_objects": {},
                    "gravity": tuple(float(x) for x in model.opt.gravity),
                    "gravity_forced": gravity_forced,
                    "steps_run": steps_run,
                    "stable_frames": stable,
                    "mapped_objects": sorted(list(obj_qposadr.keys())),
                    "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
                }

            err = _check_workspace()
            if err is not None:
                err["step"] = t + 1
                if gravity_forced:
                    model.opt.gravity[:] = orig_gravity
                return {
                    "ok": False,
                    "errors": [err],
                    "mode": "mujoco_physics",
                    "phase": "settle",
                    "settled_objects": {},
                    "gravity": tuple(float(x) for x in model.opt.gravity),
                    "gravity_forced": gravity_forced,
                    "steps_run": steps_run,
                    "stable_frames": stable,
                    "mapped_objects": sorted(list(obj_qposadr.keys())),
                    "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
                }

            # Stability: look only at free-joint dof velocities for objects.
            # If we failed to map any free joints, don't early-exit.
            if obj_dofadr:
                vmax = 0.0
                for _, dadr in obj_dofadr.items():
                    vv = data.qvel[dadr : dadr + 6]
                    vmax = max(vmax, float(abs(vv).max()))
                if vmax < float(stop_vel):
                    stable += 1
                    if stable >= int(stop_frames):
                        break
                else:
                    stable = 0

        settled_objects: dict[str, dict] = {}
        for name, adr in obj_qposadr.items():
            pos = [float(data.qpos[adr + 0]), float(data.qpos[adr + 1]), float(data.qpos[adr + 2])]
            rot = [
                float(data.qpos[adr + 3]),
                float(data.qpos[adr + 4]),
                float(data.qpos[adr + 5]),
                float(data.qpos[adr + 6]),
            ]
            settled_objects[name] = {"pos": pos, "rot": rot}

        report = {
            "ok": True,
            "errors": [],
            "mode": "mujoco_physics",
            "phase": "ok",
            "settled_objects": settled_objects,
            "gravity": tuple(float(x) for x in model.opt.gravity),
            "gravity_forced": gravity_forced,
            "steps_run": steps_run,
            "stable_frames": stable,
            "mapped_objects": sorted(list(obj_qposadr.keys())),
            "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
        }
        if gravity_forced:
            model.opt.gravity[:] = orig_gravity
        return report

    def _validate_with_incremental_insertion(
        self,
        *,
        objects: list[dict],
        robots: list[dict],
        workspace: dict,
        mujoco_assets: dict[str, dict],
        involved_objects: list[str],
        decor_objects: list[str],
        debug: bool,
        steps: int,
        pen_tol: float,
        deep_pen: float,
        ground_pen_ok: float,
        eps_z: float,
        per_obj_attempts: int,
        settle_stop_vel: float,
        settle_stop_frames: int,
    ) -> dict:
        import mujoco

        with _filter_stderr_prefixes():
            # Match the default "human-check" MuJoCo env: no scene -> handler adds its own default ground plane.
            self._ensure_mujoco_handler(mujoco_assets=mujoco_assets, robots=robots, scene=None)
            handler = self._handler
            assert handler is not None

            # IMPORTANT: use the same dm_control physics model/data that tasks use (human-check/replay),
            # not the separately-exported native model, to avoid pose mismatches.
            phy = getattr(handler, "physics", None)
            if phy is None or getattr(getattr(phy, "model", None), "ptr", None) is None or getattr(getattr(phy, "data", None), "ptr", None) is None:
                raise RuntimeError("MuJoCo dm_control physics not available on handler.")
            model = phy.model.ptr
            data = phy.data.ptr

            # Ensure gravity is enabled for settle.
            orig_gravity = tuple(float(x) for x in model.opt.gravity)
            gravity_forced = False
            if abs(orig_gravity[0]) + abs(orig_gravity[1]) + abs(orig_gravity[2]) < 1e-8:
                model.opt.gravity[:] = [0.0, 0.0, -9.81]
                gravity_forced = True

            mujoco.mj_resetData(model, data)
            data.qvel[:] = 0
            if hasattr(data, "ctrl") and data.ctrl is not None and data.ctrl.size > 0:
                data.ctrl[:] = 0

            obj_prefix = {name: self._owner_prefix[name] for name in mujoco_assets.keys() if name in self._owner_prefix}
            robot_prefix = {r["name"]: self._robot_prefix[r["name"]] for r in robots if r.get("name") in self._robot_prefix}

            def _find_free_joint_id(prefix: str) -> int | None:
                p0 = prefix.rstrip("/")
                for jid in range(model.njnt):
                    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                        continue
                    try:
                        body_id = int(model.jnt_bodyid[jid])
                        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    except Exception:
                        bname = None
                    if bname and (bname.startswith(prefix) or bname.startswith(p0)):
                        return jid
                    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and (jname.startswith(prefix) or jname.startswith(p0)):
                        return jid
                return None

            obj_qposadr: dict[str, int] = {}
            obj_dofadr: dict[str, int] = {}
            for name, prefix in obj_prefix.items():
                jid = _find_free_joint_id(prefix)
                if jid is None:
                    continue
                obj_qposadr[name] = int(model.jnt_qposadr[jid])
                obj_dofadr[name] = int(model.jnt_dofadr[jid])

            # Object geoms by ownership.
            #
            # IMPORTANT: Do not rely on geom name prefixes alone.
            # Some MJCFs name geoms generically, while bodies/joints carry the unique prefix.
            # If we miss table geoms here, tabletop insertion will mis-classify table contacts as
            # object-object collisions and will never lift Z on support penetration.
            obj_geoms: dict[str, list[int]] = {n: [] for n in obj_prefix.keys()}
            for gid in range(model.ngeom):
                try:
                    body_id = int(model.geom_bodyid[gid])
                    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                except Exception:
                    bname = ""
                owner = self._owner_from_body_name(bname) if bname else None
                if owner in obj_geoms:
                    obj_geoms[owner].append(gid)
                    continue
                # Fallback: prefix match on geom name (works for some assets).
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                for name, prefix in obj_prefix.items():
                    if gname.startswith(prefix):
                        obj_geoms[name].append(gid)
                        break

            # Robot geoms by prefix.
            robot_geom_ids: set[int] = set()
            for gid in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                for _, prefix in robot_prefix.items():
                    if gname.startswith(prefix):
                        robot_geom_ids.add(gid)

            # Identify ground/table geoms (best-effort).
            ground_geom_ids: set[int] = set()
            table_geom_ids: set[int] = set()
            for gid in range(model.ngeom):
                if gid in robot_geom_ids:
                    continue
                gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
                gtype = int(model.geom_type[gid])
                if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE) or "floor" in gname or "ground" in gname:
                    ground_geom_ids.add(gid)
                if "table" in gname or "counter" in gname or "desk" in gname:
                    table_geom_ids.add(gid)

            # Teleport all objects to isolation.
            iso = [100.0, 100.0, -10.0]
            for name, adr in obj_qposadr.items():
                data.qpos[adr : adr + 3] = iso
            if debug:
                print(f"  🧱 [insert] Isolation zone: pos={iso} (objects not yet inserted)")

            # Apply robot root pose if possible, and dof_pos.
            for r in robots:
                rname = r.get("name")
                if not isinstance(rname, str) or rname not in robot_prefix:
                    continue
                prefix = robot_prefix[rname]
                rjid = _find_free_joint_id(prefix)
                if rjid is not None:
                    adr = int(model.jnt_qposadr[rjid])
                    pos = r.get("pos")
                    rot = r.get("rot")
                    if isinstance(pos, (list, tuple)) and len(pos) == 3:
                        data.qpos[adr : adr + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
                    if isinstance(rot, (list, tuple)) and len(rot) == 4:
                        data.qpos[adr + 3 : adr + 7] = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
                dof_pos = r.get("dof_pos") or {}
                if isinstance(dof_pos, dict):
                    for jn, jv in dof_pos.items():
                        try:
                            full = f"{prefix}{jn}"
                            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
                            qadr = int(model.jnt_qposadr[jid])
                            data.qpos[qadr] = float(jv)
                        except Exception:
                            continue

            mujoco.mj_forward(model, data)

            # Determine support surface and its height (best-effort).
            ground_z: float | None = None

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
                return z0

            if ground_geom_ids:
                ground_z = max(float(data.geom_xpos[gid][2]) for gid in ground_geom_ids)

            # Choose support surface.
            surface = str((workspace.get("surface") or "")).lower()
            support_object_name = workspace.get("support_object_name")
            if not isinstance(support_object_name, str) or not support_object_name:
                support_object_name = None
            support_object_tabletop_height = workspace.get("support_object_tabletop_height")
            if not isinstance(support_object_tabletop_height, (int, float)):
                support_object_tabletop_height = None
            # Height is authored at asset scale=1; apply the table's runtime Z scale if available.
            support_object_scale_z = 1.0
            if support_object_name and support_object_name in mujoco_assets:
                sraw = (mujoco_assets.get(support_object_name) or {}).get("scale")
                if isinstance(sraw, (list, tuple)) and len(sraw) == 3 and all(isinstance(v, (int, float)) for v in sraw):
                    support_object_scale_z = float(sraw[2])
                elif isinstance(sraw, (int, float)):
                    support_object_scale_z = float(sraw)

            use_table_scene = surface == "tabletop" and bool(table_geom_ids)
            support_geom_ids = set(ground_geom_ids)
            support_kind = "ground"
            support_z = ground_z if ground_z is not None else 0.0
            if support_object_name:
                # Start with ground support; we'll switch to the support object once it is inserted.
                support_kind = "ground"
            elif use_table_scene:
                support_geom_ids = set(table_geom_ids)
                support_kind = "table"
                support_z = max(_geom_top_z(gid) for gid in table_geom_ids)

            if debug:
                print("  🪵 [insert] Z policy: z = support_z + 0.03; if colliding with support, dz += 0.01 until separated.")
                print(
                    f"  🪵 [insert] Support: kind={support_kind}, support_z={float(support_z):.3f}, dz_cap={float(self.MAX_Z_ABOVE_SUPPORT):.3f}"
                )

            by_name = {o["name"]: o for o in objects if isinstance(o, dict) and isinstance(o.get("name"), str)}

            # Insertion order: involved then decor.
            order = list(involved_objects) + list(decor_objects)
            if support_object_name and support_object_name in order:
                order = [support_object_name] + [n for n in order if n != support_object_name]
            inserted: set[str] = set()
            if debug:
                print(f"  🧩 [insert] Insertion order: involved={len(involved_objects)}, decor={len(decor_objects)}")

            def _contacts_involving(gids: set[int]) -> list[tuple[int, int, float]]:
                out = []
                for i in range(int(data.ncon)):
                    con = data.contact[i]
                    g1 = int(con.geom1)
                    g2 = int(con.geom2)
                    if g1 in gids or g2 in gids:
                        out.append((g1, g2, float(con.dist)))
                return out

            def _obj_name_from_geom(gid: int) -> str | None:
                body_id = int(model.geom_bodyid[gid])
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                return self._owner_from_body_name(bname)

            def _check_new_object(name: str) -> dict | None:
                gids = set(obj_geoms.get(name) or [])
                if not gids:
                    return {"type": "no_geoms", "obj": name}
                for g1, g2, dist in _contacts_involving(gids):
                    other = g2 if g1 in gids else g1
                    if other in robot_geom_ids:
                        continue
                    # Ground contact is allowed, but deeper penetration than COLLISION_DIST is not.
                    if other in support_geom_ids:
                        if dist < float(self.COLLISION_DIST):
                            return {
                                "type": "support_penetration",
                                "obj": name,
                                "dist": dist,
                                "kinds": [support_kind, "object"],
                                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                            }
                        continue
                    # Other object?
                    other_obj = _obj_name_from_geom(other)
                    # If the "other object" is the configured support object (table), treat it as support.
                    # This is a safety net in case support_geom_ids is incomplete due to naming quirks.
                    if support_object_name and other_obj == support_object_name and name != support_object_name:
                        if dist < float(self.COLLISION_DIST):
                            return {"type": "support_penetration", "obj": name, "dist": dist, "kinds": ["table", "object"]}
                        continue
                    if other_obj and other_obj != name and other_obj in inserted and dist < float(self.COLLISION_DIST):
                        return {"type": "collision", "obj": name, "with": other_obj, "dist": dist}
                return None

            def _set_obj_pose(name: str, x: float, y: float, z: float, yaw_rot: list[float], z_max: float) -> float | None:
                adr = obj_qposadr.get(name)
                if adr is None:
                    return None
                z_req = float(z)
                if z_req > float(z_max):
                    return None
                z = z_req
                data.qpos[adr : adr + 3] = [float(x), float(y), float(z)]
                data.qpos[adr + 3 : adr + 7] = [float(v) for v in yaw_rot]
                data.qvel[obj_dofadr.get(name, 0) : obj_dofadr.get(name, 0) + 6] = 0
                return z

            spiral = _spiral_offsets(step=0.03, turns=16, points_per_turn=18)

            inserted_z: dict[str, float] = {}
            for name in order:
                # If a support object is specified, place it on ground first, then use it as support for all others.
                saved_support: tuple[set[int], str, float] | None = None
                support_ready = bool(support_object_name is None)
                if support_object_name and name == support_object_name and not support_ready:
                    saved_support = (set(support_geom_ids), str(support_kind), float(support_z))
                    support_geom_ids = set(ground_geom_ids)
                    support_kind = "ground"
                    support_z = ground_z if ground_z is not None else 0.0

                if name not in obj_qposadr:
                    if gravity_forced:
                        model.opt.gravity[:] = orig_gravity
                    return {
                        "ok": False,
                        "errors": [{"type": "unmapped_object", "obj": name}],
                        "mode": "mujoco_incremental",
                        "phase": "pre",
                        "mapped_objects": sorted(list(obj_qposadr.keys())),
                        "unmapped_objects": sorted([n for n in obj_prefix.keys() if n not in obj_qposadr]),
                    }

                spec = by_name.get(name) or {}
                pos = spec.get("pos") or [0.0, 0.0, 0.0]
                rot = spec.get("rot") or [1.0, 0.0, 0.0, 0.0]
                x0, y0 = float(pos[0]), float(pos[1])
                # keep yaw-only quat (already computed by caller); use as-is
                yaw_rot = rot

                if debug:
                    print(f"  ➕ [insert] Trying to insert `{name}` at xy=({x0:.3f},{y0:.3f}) (max_attempts={per_obj_attempts})")

                placed_ok = False
                last_err: dict | None = None
                for attempt in range(per_obj_attempts):
                    dx, dy = spiral[attempt % len(spiral)]
                    x = _clamp(x0 + dx, float(workspace["xmin"]), float(workspace["xmax"]))
                    y = _clamp(y0 + dy, float(workspace["ymin"]), float(workspace["ymax"]))

                    # Z policy caps for this attempt (used by both fast-path and incremental lifting).
                    z0 = float(support_z) + float(self.Z_START_ABOVE_SUPPORT)
                    z_step = float(self.Z_STEP_ABOVE_SUPPORT)
                    z_max = float(support_z) + float(self.MAX_Z_ABOVE_SUPPORT)
                    # When inserting the designated support object (e.g., a table asset), its authored origin can
                    # be far from the lowest collision geometry (e.g., origin at tabletop, legs extend down).
                    # In that case, the default z cap can be too low and causes repeated attempts with no progress.
                    if saved_support is not None and support_object_name and name == support_object_name:
                        z_max = float(support_z) + max(float(self.MAX_Z_ABOVE_SUPPORT), 5.0)

                    # Special-case: inserting the support object itself (table). Many table MJCFs have an origin
                    # far from the lowest collision geometry (legs). Iterative +0.01 lifting can be very slow and
                    # can appear to "loop" across attempts. Instead, estimate the required z offset from geometry
                    # bounds (using rbound as a conservative radius) and place in one shot.
                    if saved_support is not None and support_object_name and name == support_object_name:
                        gids = set(obj_geoms.get(name) or [])
                        base_support_z = float(support_z)
                        # Start at base support and compute a conservative bottom bound.
                        z_try0 = base_support_z
                        z_used0 = _set_obj_pose(name, x, y, z_try0, yaw_rot, z_max)
                        if z_used0 is None:
                            last_err = {
                                "type": "z_cap_too_low",
                                "obj": name,
                                "required_z": z_try0,
                                "max_z": z_max,
                                "support_z": float(support_z),
                            }
                            continue
                        mujoco.mj_forward(model, data)
                        if gids:
                            min_bottom = None
                            for gid in gids:
                                try:
                                    cz = float(data.geom_xpos[gid][2])
                                    rb = float(model.geom_rbound[gid])
                                except Exception:
                                    continue
                                bottom = cz - rb
                                if min_bottom is None or bottom < min_bottom:
                                    min_bottom = bottom
                            if min_bottom is not None:
                                target_bottom = base_support_z + 1e-3
                                dz_need = float(target_bottom) - float(min_bottom)
                                if dz_need < 0.0:
                                    dz_need = 0.0
                                z_try1 = float(z_try0) + float(dz_need)
                                z_used1 = _set_obj_pose(name, x, y, z_try1, yaw_rot, z_max)
                                if z_used1 is None:
                                    last_err = {
                                        "type": "z_cap_too_low",
                                        "obj": name,
                                        "required_z": z_try1,
                                        "max_z": z_max,
                                        "support_z": float(support_z),
                                    }
                                    continue
                                mujoco.mj_forward(model, data)
                                if debug and dz_need > 1e-6:
                                    print(f"  🪵 [insert] `{name}` fast-place: dz_need≈{dz_need:.3f} -> z≈{z_try1:.3f}")

                        err = _check_new_object(name)
                        if err is None:
                            placed_ok = True
                            inserted.add(name)
                            inserted_z[name] = float(data.qpos[obj_qposadr[name] + 2])
                            if debug:
                                z_used = float(inserted_z[name])
                                dz_rel = float(z_used) - float(support_z)
                                print(
                                    f"  ✅ [insert] Inserted `{name}` at xy=({x:.3f},{y:.3f}), z={z_used:.3f} (dz={dz_rel:.3f}) after {attempt+1} attempt(s)"
                                )
                            break
                        last_err = err
                        if debug and attempt in {0, 1, 2, 4, 7, 11, 19, 39, per_obj_attempts - 1}:
                            z_used = float(data.qpos[obj_qposadr[name] + 2])
                            dz_rel = float(z_used) - float(support_z)
                            print(f"  ⚠️  [insert] `{name}` attempt {attempt+1}/{per_obj_attempts} failed, z={z_used:.3f} (dz={dz_rel:.3f}): {err}")
                        continue

                    # Z policy: start at support_z + 0.03 and lift by dz until separated from support.
                    z_used = None
                    dz = 0.0
                    while True:
                        z_try = z0 + float(dz)
                        z_used = _set_obj_pose(name, x, y, z_try, yaw_rot, z_max)
                        if z_used is None:
                            last_err = {
                                "type": "z_cap_too_low",
                                "obj": name,
                                "required_z": z_try,
                                "max_z": z_max,
                                "support_z": float(support_z),
                            }
                            break
                        mujoco.mj_forward(model, data)

                        # Check only support contacts (ground/table) for this object.
                        gids = set(obj_geoms.get(name) or [])
                        support_pen = False
                        min_support_dist = None
                        for i in range(int(data.ncon)):
                            con = data.contact[i]
                            g1 = int(con.geom1)
                            g2 = int(con.geom2)
                            if g1 not in gids and g2 not in gids:
                                continue
                            other = g2 if g1 in gids else g1
                            if other in support_geom_ids:
                                d = float(con.dist)
                                if min_support_dist is None or d < min_support_dist:
                                    min_support_dist = d
                                if d < float(self.COLLISION_DIST):
                                    support_pen = True
                                    break
                        if not support_pen:
                            break
                        # Lift policy:
                        # - default: +0.01m increments
                        # - support object (table): jump by the measured penetration depth to quickly clear ground
                        md = float(min_support_dist) if min_support_dist is not None else float("nan")
                        if saved_support is not None and support_object_name and name == support_object_name and math.isfinite(md):
                            # Need md >= COLLISION_DIST to be considered separated.
                            # If md is -0.16 and COLLISION_DIST is -0.02, jump by ~0.14 (+eps).
                            lift = max(float(z_step), float(self.COLLISION_DIST) - md + 1e-3)
                        else:
                            lift = float(z_step)
                        dz = float(dz) + float(lift)
                        if debug:
                            dz_rel = (float(z_try) - float(support_z)) + float(lift)
                            print(
                                f"  ⬆️  [insert] `{name}` support collision (min_dist={md:.4f}) -> dz += {lift:.3f} (now dz≈{dz_rel:.3f}, z≈{float(z_try) + float(lift):.3f})"
                            )
                        if z0 + float(dz) > float(z_max):
                            last_err = {
                                "type": "z_cap_too_low",
                                "obj": name,
                                "required_z": z0 + float(dz),
                                "max_z": z_max,
                                "support_z": float(support_z),
                            }
                            break

                    if last_err and last_err.get("type") == "z_cap_too_low":
                        # Can't resolve at this XY without exceeding dz cap; try another XY.
                        continue

                    # Now check full new-object constraints (object-object, deep penetrations, etc.).
                    err = _check_new_object(name)
                    if err is None:
                        placed_ok = True
                        inserted.add(name)
                        if z_used is not None:
                            inserted_z[name] = float(z_used)
                        if debug:
                            z_msg = ""
                            if z_used is not None:
                                dz_rel = float(z_used) - float(support_z)
                                z_msg = f", z={float(z_used):.3f} (dz={dz_rel:.3f})"
                            print(
                                f"  ✅ [insert] Inserted `{name}` at xy=({x:.3f},{y:.3f}){z_msg} after {attempt+1} attempt(s)"
                            )
                        break
                    last_err = err
                    if debug and attempt in {0, 1, 2, 4, 7, 11, 19, 39, per_obj_attempts - 1}:
                        z_msg = ""
                        if z_used is not None:
                            dz_rel = float(z_used) - float(support_z)
                            z_msg = f", z={float(z_used):.3f} (dz={dz_rel:.3f})"
                        print(f"  ⚠️  [insert] `{name}` attempt {attempt+1}/{per_obj_attempts} failed{z_msg}: {err}")

                if not placed_ok:
                    if last_err is None:
                        last_err = _check_new_object(name)
                    if debug:
                        print(f"  ❌ [insert] Failed to insert `{name}` after {per_obj_attempts} attempts. Last error: {last_err}")
                    if gravity_forced:
                        model.opt.gravity[:] = orig_gravity
                    return {
                        "ok": False,
                        "errors": [{"type": "insertion_failed", "obj": name, "details": last_err}],
                        "mode": "mujoco_incremental",
                        "phase": "insert",
                        "support_kind": support_kind,
                        "support_z": float(support_z),
                    }

                # After inserting the support object, switch support to it and compute tabletop height.
                if support_object_name and name == support_object_name and saved_support is not None:
                    sgids = set(obj_geoms.get(support_object_name) or [])
                    if sgids:
                        support_geom_ids = sgids
                        support_kind = "table"
                        # Use the larger of:
                        # - actual collision geometry top (from MuJoCo)
                        # - metadata tabletop height offset (table base z + height)
                        support_z_geom = max(_geom_top_z(gid) for gid in sgids)
                        support_z_meta = None
                        if support_object_tabletop_height is not None:
                            try:
                                # Base z from current qpos of the inserted (and grounded) table.
                                adr = obj_qposadr.get(support_object_name)
                                if adr is not None:
                                    base_z = float(data.qpos[adr + 2])
                                    support_z_meta = base_z + float(support_object_tabletop_height) * float(support_object_scale_z)
                            except Exception:
                                support_z_meta = None
                        # Prefer authored tabletop height if provided; geometry tops can include backboards/rails.
                        support_z = float(support_z_meta) if support_z_meta is not None else float(support_z_geom)
                        if debug:
                            print(
                                f"  🪵 [insert] Support switched to object `{support_object_name}` with support_z={float(support_z):.3f}"
                            )
                    else:
                        # Can't use as support; restore prior support.
                        support_geom_ids, support_kind, support_z = saved_support
                        support_object_name = None
                        if debug:
                            print(f"  ⚠️  [insert] support_object_name `{name}` has no geoms; restored previous support.")

            # Settle stage: run the physics engine headlessly and let objects fall under gravity,
            # consistent with the later "humancheck" behavior.
            if debug:
                print("  🧘 [settle] Running MuJoCo physics settle (free-fall under gravity)...")

            is_tabletop = str((workspace.get("surface") or "")).lower() == "tabletop"
            # In tabletop, free-fall can cause lateral sliding/spin and "pop" from contacts.
            # To make settle stable/deterministic for layout initialization, clamp x/y and orientation during settle
            # (z-only motion). This keeps the semantic XY layout fixed while gravity resolves vertical placement.
            settle_z_only = bool(is_tabletop)
            # Also disable robot collisions during settle so the robot doesn't interfere with settling.
            # (We only use settle to get a plausible z placement for objects/table.)
            robot_collisions_disabled = False
            robot_geom_collision_backup: dict[int, tuple[int, int]] = {}
            # Optionally "remove" robots during settle by teleporting them to an isolation zone.
            # We keep this lightweight (no model rebuild): move robot free joints far away, and restore after settle.
            settle_remove_robot = True
            robot_pose_backup: dict[str, tuple[float, float, float, float, float, float, float]] = {}
            robot_qposadr: dict[str, int] = {}
            try:
                for gid in robot_geom_ids:
                    robot_geom_collision_backup[int(gid)] = (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid]))
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = 0
                robot_collisions_disabled = bool(robot_geom_collision_backup)
            except Exception:
                robot_collisions_disabled = False

            frozen_xy_quat: dict[str, tuple[float, float, float, float, float, float]] = {}
            if settle_z_only:
                for oname in inserted:
                    adr = obj_qposadr.get(oname)
                    if adr is None:
                        continue
                    frozen_xy_quat[oname] = (
                        float(data.qpos[adr + 0]),
                        float(data.qpos[adr + 1]),
                        float(data.qpos[adr + 3]),
                        float(data.qpos[adr + 4]),
                        float(data.qpos[adr + 5]),
                        float(data.qpos[adr + 6]),
                    )
                if debug:
                    print(f"  🧘 [settle] Mode: z-only (clamp xy+quat) for {len(frozen_xy_quat)} object(s).")
                    if robot_collisions_disabled:
                        print(f"  🧘 [settle] Robot collisions disabled for {len(robot_geom_collision_backup)} geom(s).")

            # Teleport robots away for settle (optional), then restore later.
            if settle_remove_robot and robot_prefix:
                iso_r = (100.0, 100.0, -10.0)
                for r in robots:
                    rname = r.get("name")
                    if not isinstance(rname, str) or rname not in robot_prefix:
                        continue
                    prefix = robot_prefix[rname]
                    rjid = _find_free_joint_id(prefix)
                    if rjid is None:
                        continue
                    adr = int(model.jnt_qposadr[rjid])
                    robot_qposadr[rname] = adr
                    robot_pose_backup[rname] = (
                        float(data.qpos[adr + 0]),
                        float(data.qpos[adr + 1]),
                        float(data.qpos[adr + 2]),
                        float(data.qpos[adr + 3]),
                        float(data.qpos[adr + 4]),
                        float(data.qpos[adr + 5]),
                        float(data.qpos[adr + 6]),
                    )
                    data.qpos[adr + 0] = iso_r[0]
                    data.qpos[adr + 1] = iso_r[1]
                    data.qpos[adr + 2] = iso_r[2]
                    data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                if debug and robot_pose_backup:
                    print(f"  🧘 [settle] Robots teleported to isolation for settle: {sorted(robot_pose_backup.keys())}")

            def _max_object_speed() -> float:
                vmax = 0.0
                for oname in inserted:
                    dof0 = obj_dofadr.get(oname)
                    if dof0 is None:
                        continue
                    try:
                        v = data.qvel[dof0 : dof0 + 6]
                        # linear(3)+angular(3)
                        s = float(math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + float(v[3] * v[3] + v[4] * v[4] + v[5] * v[5])))
                    except Exception:
                        continue
                    vmax = max(vmax, s)
                return vmax

            stable = 0
            steps_run = 0
            try:
                render_settle = bool(debug) and str(os.environ.get("TASKGEN_RENDER_SETTLE") or "").strip() not in {"", "0", "false", "False"}
                viewer_mod = None
                if render_settle:
                    try:
                        import mujoco.viewer as viewer_mod  # type: ignore[import-not-found]
                    except Exception:
                        viewer_mod = None
                        if debug:
                            print("  ⚠️  [settle] TASKGEN_RENDER_SETTLE set, but mujoco.viewer not available; continuing headless.")

                def _run_settle_loop(viewer=None):
                    nonlocal stable, steps_run
                    for _ in range(int(steps)):
                        mujoco.mj_step(model, data)
                        steps_run += 1

                        if settle_z_only and frozen_xy_quat:
                            # Clamp x/y and quaternion; zero x/y + angular velocities so only z can change.
                            for oname, (x0, y0, qw, qx, qy, qz) in frozen_xy_quat.items():
                                adr = obj_qposadr.get(oname)
                                if adr is not None:
                                    data.qpos[adr + 0] = x0
                                    data.qpos[adr + 1] = y0
                                    data.qpos[adr + 3] = qw
                                    data.qpos[adr + 4] = qx
                                    data.qpos[adr + 5] = qy
                                    data.qpos[adr + 6] = qz
                                dof0 = obj_dofadr.get(oname)
                                if dof0 is not None:
                                    data.qvel[dof0 + 0] = 0.0
                                    data.qvel[dof0 + 1] = 0.0
                                    data.qvel[dof0 + 3] = 0.0
                                    data.qvel[dof0 + 4] = 0.0
                                    data.qvel[dof0 + 5] = 0.0
                            mujoco.mj_forward(model, data)

                        # Stop once all objects have sufficiently low velocity for a sustained window.
                        vmax = _max_object_speed()
                        if vmax <= float(settle_stop_vel):
                            stable += 1
                        else:
                            stable = 0
                        if stable >= int(settle_stop_frames):
                            break

                        if viewer is not None:
                            try:
                                if hasattr(viewer, "is_running") and not viewer.is_running():
                                    break
                                if hasattr(viewer, "sync"):
                                    viewer.sync()
                            except Exception:
                                pass
                            # Keep UI responsive without pegging CPU.
                            try:
                                time.sleep(float(model.opt.timestep))
                            except Exception:
                                pass

                if viewer_mod is not None:
                    # Render settle in an interactive viewer (debug only).
                    with viewer_mod.launch_passive(model, data) as viewer:
                        _run_settle_loop(viewer=viewer)
                else:
                    _run_settle_loop(viewer=None)
            finally:
                # Restore robot poses (and keep them non-colliding until after pose adjustments).
                if robot_pose_backup:
                    for rname, pose7 in robot_pose_backup.items():
                        adr = robot_qposadr.get(rname)
                        if adr is None:
                            continue
                        data.qpos[adr + 0] = float(pose7[0])
                        data.qpos[adr + 1] = float(pose7[1])
                        data.qpos[adr + 2] = float(pose7[2])
                        data.qpos[adr + 3] = float(pose7[3])
                        data.qpos[adr + 4] = float(pose7[4])
                        data.qpos[adr + 5] = float(pose7[5])
                        data.qpos[adr + 6] = float(pose7[6])
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                # Restore robot collisions.
                if robot_geom_collision_backup:
                    try:
                        for gid, (ct, ca) in robot_geom_collision_backup.items():
                            model.geom_contype[gid] = int(ct)
                            model.geom_conaffinity[gid] = int(ca)
                    except Exception:
                        pass
            if debug:
                print(f"  🧘 [settle] Done: steps_run={steps_run}, stable_frames={stable} (v_stop={settle_stop_vel})")
                try:
                    plane_zs = [float(data.geom_xpos[gid][2]) for gid in range(model.ngeom) if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
                    if plane_zs:
                        print(f"  🧱 [settle] Ground plane z range: [{min(plane_zs):.3f}, {max(plane_zs):.3f}]")
                except Exception:
                    pass

            # After settling objects in tabletop mode, align robot root z to the minimum object z (excluding table).
            # This is a visualization / convenience policy for downstream init_state.
            settled_robots: dict[str, dict] = {}
            is_tabletop = str((workspace.get("surface") or "")).lower() == "tabletop"
            if is_tabletop:
                tabletop_robot_xy = (-0.5, 0.0)
                def _find_root_body_id_by_prefix(prefix: str) -> int | None:
                    p0 = prefix.rstrip("/")
                    # Fast path: exact body name.
                    try:
                        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p0))
                        if bid >= 0:
                            return bid
                    except Exception:
                        pass

                    candidates: list[int] = []
                    for i in range(int(model.nbody)):
                        try:
                            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                        except Exception:
                            bname = None
                        if not bname:
                            continue
                        if bname == p0 or bname.startswith(p0):
                            candidates.append(int(i))

                    if not candidates:
                        return None

                    def _depth_to_world(bid: int) -> int:
                        depth = 0
                        cur = bid
                        # body 0 is the world body in MuJoCo.
                        while cur != 0 and depth < 64:
                            try:
                                cur = int(model.body_parentid[cur])
                            except Exception:
                                break
                            depth += 1
                        return depth

                    # Prefer the shallowest (closest-to-world) body; tie-break by shorter name.
                    best = None
                    best_key = None
                    for bid in candidates:
                        try:
                            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                        except Exception:
                            bname = ""
                        key = (_depth_to_world(bid), len(bname))
                        if best is None or key < best_key:
                            best = bid
                            best_key = key
                    return best

                min_obj_z = None
                for oname in inserted:
                    if support_object_name and oname == support_object_name:
                        continue
                    adr = obj_qposadr.get(oname)
                    if adr is None:
                        continue
                    z = float(data.qpos[adr + 2])
                    if min_obj_z is None or z < min_obj_z:
                        min_obj_z = z

                if min_obj_z is not None:
                    for r in robots:
                        rname = r.get("name")
                        if not isinstance(rname, str) or rname not in robot_prefix:
                            continue
                        prefix = robot_prefix[rname]
                        rjid = _find_free_joint_id(prefix)
                        if rjid is not None:
                            adr = int(model.jnt_qposadr[rjid])
                            data.qpos[adr + 0] = float(tabletop_robot_xy[0])
                            data.qpos[adr + 1] = float(tabletop_robot_xy[1])
                            data.qpos[adr + 2] = float(min_obj_z)
                            data.qvel[:] = 0
                            mujoco.mj_forward(model, data)
                            settled_robots[rname] = {
                                "pos": [
                                    float(data.qpos[adr + 0]),
                                    float(data.qpos[adr + 1]),
                                    float(data.qpos[adr + 2]),
                                ],
                                "rot": [
                                    float(data.qpos[adr + 3]),
                                    float(data.qpos[adr + 4]),
                                    float(data.qpos[adr + 5]),
                                    float(data.qpos[adr + 6]),
                                ],
                            }
                            continue

                        # Fixed-base robots don't have a free joint; align by shifting the robot root body instead.
                        body_id = _find_root_body_id_by_prefix(prefix)
                        if body_id is None:
                            continue
                        try:
                            model.body_pos[body_id, 0] = float(tabletop_robot_xy[0])
                            model.body_pos[body_id, 1] = float(tabletop_robot_xy[1])
                            model.body_pos[body_id, 2] = float(min_obj_z)
                        except Exception:
                            try:
                                model.body_pos[body_id][0] = float(tabletop_robot_xy[0])
                                model.body_pos[body_id][1] = float(tabletop_robot_xy[1])
                                model.body_pos[body_id][2] = float(min_obj_z)
                            except Exception:
                                continue
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                        try:
                            pos = [float(data.xpos[body_id][0]), float(data.xpos[body_id][1]), float(data.xpos[body_id][2])]
                            rot = [
                                float(data.xquat[body_id][0]),
                                float(data.xquat[body_id][1]),
                                float(data.xquat[body_id][2]),
                                float(data.xquat[body_id][3]),
                            ]
                        except Exception:
                            pos, rot = None, None
                        if pos is not None and rot is not None:
                            settled_robots[rname] = {"pos": pos, "rot": rot}

            settled_objects: dict[str, dict] = {}
            for name, adr in obj_qposadr.items():
                if name not in inserted:
                    continue
                settled_pos = [float(data.qpos[adr + 0]), float(data.qpos[adr + 1]), float(data.qpos[adr + 2])]
                settled_objects[name] = {
                    "pos": [float(data.qpos[adr + 0]), float(data.qpos[adr + 1]), float(data.qpos[adr + 2])],
                    "rot": [
                        float(data.qpos[adr + 3]),
                        float(data.qpos[adr + 4]),
                        float(data.qpos[adr + 5]),
                        float(data.qpos[adr + 6]),
                    ],
                }
                if debug and name in inserted_z:
                    dz = float(settled_pos[2]) - float(inserted_z[name])
                    if abs(dz) > 5e-3:
                        print(f"  🧷 [settle] `{name}` z changed by {dz:+.3f} (inserted {inserted_z[name]:.3f} -> settled {settled_pos[2]:.3f})")

            out = {
                "ok": True,
                "errors": [],
                "mode": "mujoco_incremental",
                "phase": "ok",
                "settled_objects": settled_objects,
                "settled_robots": settled_robots,
                "support_kind": support_kind,
                "support_z": float(support_z),
                "steps_run": steps_run,
                "stable_frames": stable,
            }
            if gravity_forced:
                model.opt.gravity[:] = orig_gravity
            return out


class LayoutRepairer:
    """Deterministically repair layouts (no GPT calls) via local moves + resampling."""

    def __init__(self, *, max_iterations: int = 200):
        self._max_iterations = max_iterations

    def repair(
        self,
        *,
        objects: list[dict],
        robots: list[dict],
        workspace: dict,
        min_clearance: float,
        priorities: dict[str, int],
        validator: LayoutValidator,
        mujoco_assets: dict[str, dict] | None = None,
    ) -> tuple[list[dict], dict]:
        xmin, xmax = float(workspace["xmin"]), float(workspace["xmax"])
        ymin, ymax = float(workspace["ymin"]), float(workspace["ymax"])

        obj_by_name = {o["name"]: o for o in objects}
        spiral = _spiral_offsets(step=0.02, turns=14, points_per_turn=20)

        def clamp_obj_xy(name: str) -> None:
            o = obj_by_name[name]
            x, y, z = o["pos"]
            o["pos"] = [_clamp(float(x), xmin, xmax), _clamp(float(y), ymin, ymax), float(z)]

        def nudge(name: str, k: int) -> None:
            o = obj_by_name[name]
            ox, oy, oz = o["pos"]
            dx, dy = spiral[k % len(spiral)]
            o["pos"] = [_clamp(float(ox) + dx, xmin, xmax), _clamp(float(oy) + dy, ymin, ymax), float(oz)]

        last_report = {"ok": False, "errors": []}
        for it in range(self._max_iterations):
            report = validator.validate_fast(
                objects=list(obj_by_name.values()),
                workspace=workspace,
                min_clearance=min_clearance,
            )
            last_report = report
            if report["ok"]:
                report = dict(report)
                report["iterations"] = it + 1
                return list(obj_by_name.values()), report

            for err in report["errors"]:
                if err["type"] == "out_of_bounds" and err.get("obj"):
                    clamp_obj_xy(err["obj"])
                    continue
                if err["type"] == "collision" and err.get("a") and err.get("b"):
                    a, b = err["a"], err["b"]
                    pa = priorities.get(a, 0)
                    pb = priorities.get(b, 0)
                    move = b if pb < pa else a if pa < pb else random.choice([a, b])
                    nudge(move, it)
                    continue

        last_report = dict(last_report)
        last_report["iterations"] = self._max_iterations
        return list(obj_by_name.values()), last_report


if TYPE_CHECKING:
    from roboverse_pack.tasks.gpt.gpt_base import ObjectState, TaskSpec


# ======================================
# Configuration
# ======================================


@dataclass
class PathConfig:
    """Centralized path configuration."""
    taskgen_dir: Path = Path("taskgen_json")
    tasks_output: Path = Path("metasim/cfg/tasks/gpt/config/tasks")
    pkl_output: Path = Path("roboverse_data/trajs/gpt")
    task_py_output: Path = Path("roboverse_pack/tasks/gpt")

    @property
    def object_registry(self) -> Path:
        return self.taskgen_dir / "category_registry_objects.json"

    @property
    def robot_registry(self) -> Path:
        return self.taskgen_dir / "category_registry_robots.json"


@dataclass
class GPTConfig:
    """GPT API configuration."""
    model: str = "gpt-4o-2024-08-06"  # TODO: change to official setting
    base_url: str = "https://yunwu.ai/v1"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 15000
    difficulty: int = 3

    @classmethod
    def from_env(cls, difficulty: int = 3) -> GPTConfig:
        return cls(api_key=os.getenv("OPENAI_API_KEY"), difficulty=difficulty)


# ======================================
# Task Generator
# ======================================


class TaskGenerator:
    """Three-stage task generation pipeline."""

    def __init__(self, client: openai.OpenAI, paths: PathConfig):
        self._client = client
        self._paths = paths
        self._obj_registry: dict | None = None
        self._robot_registry: dict | None = None
        self._gpt_cfg = GPTConfig.from_env()
        self._layout_validator = LayoutValidator(use_mujoco=True)
        self._layout_repairer = LayoutRepairer(max_iterations=200)
        self._last_raw_by_stage: dict[str, str] = {}

    @property
    def obj_registry(self) -> dict:
        if self._obj_registry is None:
            self._obj_registry = self._load_json(self._paths.object_registry)
        return self._obj_registry

    @property
    def robot_registry(self) -> dict:
        if self._robot_registry is None:
            self._robot_registry = self._load_json(self._paths.robot_registry)
        return self._robot_registry

    def _load_json(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _call_gpt(self, system: str, user: str = "", debug_stage: str | None = None) -> dict | None:
        messages = [{"role": "system", "content": system}]
        if user:
            messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model=self._gpt_cfg.model,
            messages=messages,
            temperature=self._gpt_cfg.temperature,
            max_tokens=self._gpt_cfg.max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = self._parse_json_response(content)
        if parsed is None and debug_stage:
            self._last_raw_by_stage[debug_stage] = content
            preview = content if len(content) <= 6000 else (content[:6000] + "\n...[truncated]...")
            print(f"\n  ⚠️  [{debug_stage}] Failed to parse JSON from model response. Raw output:\n{preview}\n")
        return parsed

    def _extract_json(self, content: str) -> str:
        """Extract JSON from potential markdown code block."""
        if content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content[3:]
            if content.endswith("```"):
                content = content[:-3]
        return content.strip()

    def _parse_json_response(self, content: str) -> dict | None:
        """Parse a model response that should contain a single JSON object.

        Be tolerant to:
        - markdown fences
        - leading/trailing prose
        - trailing commas
        """
        raw = content.strip()
        s = self._extract_json(raw)

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

        for cand in (s, _strip_to_braces(s), _remove_trailing_commas(s), _remove_trailing_commas(_strip_to_braces(s))):
            try:
                obj = json.loads(cand)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                continue
        return None

    def generate(self, user_prompt: str, difficulty: int = 3) -> tuple[TaskSpec | None, dict]:
        """Main entry: three-stage generation pipeline.

        Returns:
            (TaskSpec or None, reasoning_metadata dict)
        """
        reasoning_metadata = {"difficulty": difficulty, "timestamp": datetime.now().isoformat()}

        # Stage 1
        TaskGenUI.print_stage_start(1, "🎯 Stage 1: Selecting categories...")
        stage1_failures: list[str] = []

        def _stage1_feedback() -> str:
            if not stage1_failures:
                return "(none)"
            return "\n".join([f"- {s}" for s in stage1_failures[-4:]])

        categories = None
        for attempt in range(4):
            categories = self._stage1_select_categories(user_prompt, failure_feedback=_stage1_feedback())
            if not categories:
                stage1_failures.append("gpt:null_or_parse_failed")
                continue
            obj_cats = categories.get("object_categories")
            if not isinstance(obj_cats, list) or any(not isinstance(x, str) for x in obj_cats) or not obj_cats:
                stage1_failures.append("schema:object_categories_invalid")
                categories = None
                continue
            allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
            if allow_table and len(obj_cats) == 1 and obj_cats[0] == "table":
                # This almost always breaks Stage2 (no manipulable objects). Force a retry.
                stage1_failures.append("invalid:only_table_selected")
                categories = None
                continue
            if allow_table and "table" not in obj_cats:
                stage1_failures.append("invalid:allow_table_requires_table_category")
                categories = None
                continue
            if allow_table and len(obj_cats) < 2:
                stage1_failures.append("invalid:allow_table_requires_non_table_category")
                categories = None
                continue

            # Ensure Stage1 picks enough objects to satisfy Stage2 decorative requirement.
            try:
                need = int(DIFFICULTY_DEFINITIONS[difficulty]["objects_decorative"]) + (2 if allow_table else 1)
            except Exception:
                need = 2
            try:
                uniq: set[str] = set()
                for c in obj_cats:
                    entry = (self.obj_registry.get("categories") or {}).get(c) or {}
                    names = entry.get("object_names") or []
                    if isinstance(names, list):
                        for n in names:
                            if isinstance(n, str):
                                uniq.add(n)
                if len(uniq) < need:
                    stage1_failures.append(f"invalid:too_few_objects_available:{len(uniq)}<{need}")
                    categories = None
                    continue
            except Exception:
                # If we can't estimate, don't block generation.
                pass
            break

        if not categories:
            reasoning_metadata["stage1_failures"] = stage1_failures
            TaskGenUI.print_error("GPT returned None. Try a more specific task description.")
            return None, reasoning_metadata
        reasoning_metadata["stage1_reasoning"] = categories.get("reasoning", {})
        print(f"  📦 Object categories: {categories['object_categories']}")
        print(f"  🤖 Robot category: {categories['robot_category']}\n")

        # Load assets
        all_objs = self._load_objects(categories["object_categories"])
        all_robots = self._load_robots([categories["robot_category"]])
        print(f"  ✅ Loaded {len(all_objs)} objects, {len(all_robots)} robots\n")
        allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
        table_object_names: set[str] = set()
        if allow_table:
            entry = (self.obj_registry.get("categories") or {}).get("table") or {}
            names = entry.get("object_names") or []
            if isinstance(names, list):
                table_object_names = {str(n) for n in names if isinstance(n, str)}
            if not table_object_names:
                TaskGenUI.print_error("TASKGEN_ALLOW_TABLE=1 but category `table` has no object_names in registry.")
                reasoning_metadata["stage1_error"] = "allow_table_table_category_empty"
                return None, reasoning_metadata
            if not (table_object_names & set(all_objs.keys())):
                TaskGenUI.print_error(
                    "TASKGEN_ALLOW_TABLE=1 but no table-category objects were loaded. Ensure Stage1 includes category `table` and its detail file is valid."
                )
                reasoning_metadata["stage1_error"] = "allow_table_missing_table_objects"
                return None, reasoning_metadata

        # Stage 2
        TaskGenUI.print_stage_start(2, "📝 Stage 2: Generating task...")
        task_concept = categories.get("task_concept", "")
        partial = None
        stage2_failures: list[str] = []

        def _stage2_feedback() -> str:
            if not stage2_failures:
                return "(none)"
            return "\n".join([f"- {s}" for s in stage2_failures[-4:]])

        for attempt in range(6):
            cand = self._stage2_generate_task(
                user_prompt,
                list(all_objs.keys()),
                list(all_robots.keys()),
                task_concept,
                difficulty,
                failure_feedback=_stage2_feedback(),
            )
            if not cand:
                if self._last_raw_by_stage.get("stage2"):
                    reasoning_metadata["stage2_raw"] = self._last_raw_by_stage["stage2"]
                stage2_failures.append("gpt:null_or_parse_failed")
                continue
            # Hard constraints for Stage2 outputs.
            objects_involved = cand.get("objects_involved") or []
            decor_objects = cand.get("decor_objects") or []
            robot_involved = cand.get("robot_involved") or []
            if not isinstance(objects_involved, list) or not isinstance(decor_objects, list):
                stage2_failures.append("schema:objects_involved_or_decor_not_list")
                continue
            if any(not isinstance(x, str) for x in objects_involved + decor_objects):
                stage2_failures.append("schema:object_names_not_str")
                continue
            if not isinstance(robot_involved, list) or any(not isinstance(x, str) for x in robot_involved):
                stage2_failures.append("schema:robot_involved_not_list_or_not_str")
                continue
            if any(x not in all_objs for x in objects_involved + decor_objects):
                bad = [x for x in objects_involved + decor_objects if x not in all_objs][:6]
                stage2_failures.append(f"invalid_object_names:{bad}")
                continue
            if any(x not in all_robots for x in robot_involved):
                bad = [x for x in robot_involved if x not in all_robots][:6]
                stage2_failures.append(f"invalid_robot_names:{bad}")
                continue
            allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
            if allow_table:
                table_involved = [n for n in objects_involved if n in table_object_names]
                table_in_decor = [n for n in decor_objects if n in table_object_names]
                if table_in_decor:
                    stage2_failures.append(f"invalid:table_in_decor:{table_in_decor[:3]}")
                    continue
                if len(table_involved) != 1:
                    stage2_failures.append(f"invalid:need_exactly_one_table_in_involved:got:{table_involved}")
                    continue
            else:
                if any(n in table_object_names for n in objects_involved + decor_objects):
                    stage2_failures.append("invalid:table_selected_when_disallowed")
                    continue
            if len(set(objects_involved)) != len(objects_involved):
                stage2_failures.append("set:duplicate_objects_involved")
                continue
            if len(set(decor_objects)) != len(decor_objects):
                stage2_failures.append("set:duplicate_decor_objects")
                continue
            if set(objects_involved) & set(decor_objects):
                stage2_failures.append("set:involved_overlaps_decor")
                continue
            if len(decor_objects) != DIFFICULTY_DEFINITIONS[difficulty]["objects_decorative"]:
                stage2_failures.append(
                    f"decor_count_expected:{DIFFICULTY_DEFINITIONS[difficulty]['objects_decorative']}_got:{len(decor_objects)}"
                )
                continue
            partial = cand
            break
        if not partial:
            reasoning_metadata["stage2_failures"] = stage2_failures
            TaskGenUI.print_error("GPT returned None. Try a more specific task description.")
            return None, reasoning_metadata
        reasoning_metadata["stage2_reasoning"] = partial.get("reasoning", {})
        print(f"  🏷️  Task: {partial['task_name']}\n")

        # Stage 3
        TaskGenUI.print_stage_start(3, "🗺️  Stage 3: Layout (GPT + validate/repair)...")
        spec, layout = self._stage3_generate_layout(partial, all_objs, all_robots, difficulty)
        if not spec:
            if isinstance(layout, dict) and layout.get("error_type") == "validation_failed":
                TaskGenUI.print_error("Layout validation failed. See `stage3_error` in generation metadata.")
                reasoning_metadata["stage3_error"] = layout
            else:
                TaskGenUI.print_error("GPT returned None. Try a more specific task description.")
            return None, reasoning_metadata
        if layout:
            reasoning_metadata["stage3_reasoning"] = layout.get("reasoning", {})
        print("  ✅ Layout generated\n")
        TaskGenUI.print_layout(spec)

        return spec, reasoning_metadata

    def _stage1_select_categories(self, user_prompt: str, *, failure_feedback: str = "") -> dict:
        allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
        obj_cats = {
            k: {"desc": v["category_description"], "objects": v["object_names"]}
            for k, v in self.obj_registry["categories"].items()
            if v.get("object_count", 0) > 0
        }
        if not allow_table:
            obj_cats.pop("table", None)
        robot_cats = {
            k: {"desc": v["category_description"], "robots": v["robot_names"]}
            for k, v in self.robot_registry["categories"].items()
            if v.get("robot_count", 0) > 0
        }
        system = format_stage1_prompt(obj_cats, robot_cats, allow_table=allow_table)
        # Append failures as user message suffix to encourage self-correction without changing schema.
        if failure_feedback and failure_feedback.strip() and failure_feedback.strip() != "(none)":
            user_prompt = user_prompt.strip() + "\n\nPrevious failures:\n" + failure_feedback.strip()
        return self._call_gpt(system, user_prompt)

    def _stage2_generate_task(
        self,
        user_prompt: str,
        object_list: list,
        robot_list: list,
        task_concept: str,
        difficulty: int,
        failure_feedback: str = "",
    ) -> dict:
        allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
        table_object_names: list[str] = []
        try:
            if allow_table:
                entry = (self.obj_registry.get("categories") or {}).get("table") or {}
                names = entry.get("object_names") or []
                if isinstance(names, list):
                    table_object_names = [str(n) for n in names if isinstance(n, str)]
        except Exception:
            table_object_names = []

        system = format_stage2_prompt(
            object_list,
            robot_list,
            task_concept,
            difficulty,
            allow_table=allow_table,
            table_object_names=table_object_names,
            failure_feedback=failure_feedback,
        )
        return self._call_gpt(system, user_prompt, debug_stage="stage2")

    def _stage3_generate_layout(
        self, partial: dict, all_objs: dict, all_robots: dict, difficulty: int
    ) -> tuple[TaskSpec | None, dict | None]:
        """Generate layout and return (TaskSpec, layout_with_reasoning).

        Stage 3 now asks GPT for an intent + coarse layout (XY + yaw), then we deterministically
        finalize poses and repair collisions/bounds via a validator/repairer loop.
        """
        from roboverse_pack.tasks.gpt.gpt_base import ObjectState, Position, RobotState, Rotation, TaskSpec

        condensed_objs = {
            name: {"z": data["init_state"]["pos"][2], "rot": data["init_state"]["rot"]}
            for name, data in all_objs.items()
        }
        condensed_robots = {
            name: {"pos": data["pos"], "rot": data["rot"], "dof_pos": data["dof_pos"]}
            for name, data in all_robots.items()
        }

        involved_list = list(partial.get("objects_involved") or [])
        decor_list = list(partial.get("decor_objects") or [])
        target_set = set(involved_list) | set(decor_list)
        if not involved_list or len(target_set) != (len(involved_list) + len(decor_list)):
            return None, None

        def _as_xy(v) -> tuple[float, float] | None:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return None
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return None

        def _as_float(v, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return default

        # Defaults if GPT omits controls.
        num_required = len(involved_list)
        num_decor = len(decor_list)
        num_total = num_required + num_decor
        if num_total <= 4:
            default_clearance = 0.10
        elif num_total <= 7:
            default_clearance = 0.12
        elif num_total <= 10:
            default_clearance = 0.14
        else:
            default_clearance = 0.16
        min_clearance = default_clearance

        involved_names = set(involved_list)

        def _yaw_from_quat(q: list[float]) -> float:
            # q = (w,x,y,z); yaw about Z
            try:
                w, x, y, z = [float(v) for v in q]
                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                return math.atan2(siny_cosp, cosy_cosp)
            except Exception:
                return 0.0

        def _schema_set_check(layout_plan: dict) -> tuple[bool, str]:
            objs = layout_plan.get("objects")
            if not isinstance(objs, list):
                return False, "schema:objects_not_list"
            names = []
            for o in objs:
                if not isinstance(o, dict) or "name" not in o:
                    return False, "schema:object_entry_invalid"
                if not isinstance(o["name"], str):
                    return False, "schema:object_name_not_str"
                names.append(o["name"])
            if len(names) != len(set(names)):
                return False, "set:duplicate_names"
            plan_set = set(names)
            if plan_set != target_set:
                missing = sorted(list(target_set - plan_set))[:8]
                extra = sorted(list(plan_set - target_set))[:8]
                return False, f"set:mismatch missing={missing} extra={extra}"
            return True, "ok"

        failures: list[str] = []
        max_gen_attempts = 6
        max_physics_repair_iters = 120

        def _feedback() -> str:
            if not failures:
                return "(none)"
            lines = [f"- {s}" for s in failures[-3:]]
            return "\n".join(lines)

        for gen_attempt in range(1, max_gen_attempts + 1):
            print(f"  🧭 3.1 Generating coarse semantic layout (GPT) [{gen_attempt}/{max_gen_attempts}]...")
            system = format_stage3_prompt(partial, condensed_objs, condensed_robots, difficulty, failure_feedback=_feedback())
            layout_plan = self._call_gpt(system)
            if not layout_plan or not isinstance(layout_plan, dict):
                print("  ❌ 3.1a Pre-validate failed: gpt:null_or_non_dict")
                failures.append("gpt:null_or_non_dict")
                continue

            ok, reason = _schema_set_check(layout_plan)
            if not ok:
                print(f"  ❌ 3.1a Pre-validate failed: {reason}")
                failures.append(reason)
                continue

            workspace_raw = layout_plan.get("workspace") or {}
            if not isinstance(workspace_raw, dict):
                workspace_raw = {}
            workspace = {
                "xmin": _as_float(workspace_raw.get("xmin", -0.5), -0.5),
                "xmax": _as_float(workspace_raw.get("xmax", 0.5), 0.5),
                "ymin": _as_float(workspace_raw.get("ymin", -0.5), -0.5),
                "ymax": _as_float(workspace_raw.get("ymax", 0.5), 0.5),
                "surface": (workspace_raw.get("surface") or "ground"),
                "table_body_or_geom_name": (workspace_raw.get("table_body_or_geom_name") or "table"),
            }
            min_clearance = _as_float(layout_plan.get("min_clearance", default_clearance), default_clearance)

            plan_objs = layout_plan.get("objects") or []
            placements: dict[str, dict] = {
                o["name"]: o for o in plan_objs if isinstance(o, dict) and isinstance(o.get("name"), str)
            }
            missing_entries = sorted(list(target_set - set(placements.keys())))
            if missing_entries:
                # Shouldn't happen due to set check; keep as a defensive debug signal.
                print(f"  ⚠️  3.1b Missing placement entries after set check: {missing_entries[:8]}")
            missing_xy = [n for n, meta in placements.items() if _as_xy(meta.get("xy")) is None]

            print(
                f"  📐 3.2 Finalizing poses: involved={len(involved_list)}, decor={len(decor_list)}, total={len(target_set)}, min_clearance≈{min_clearance:.2f}m"
            )
            if decor_list:
                preview = ", ".join(decor_list[:6]) + ("..." if len(decor_list) > 6 else "")
                print(f"  🎨 3.2a Decorative objects (fixed): {len(decor_list)} ({preview})")
            if missing_xy:
                preview = ", ".join(missing_xy[:6]) + ("..." if len(missing_xy) > 6 else "")
                print(f"  ⚠️  3.2b Missing/invalid `xy` from GPT for {len(missing_xy)} objects (will auto-place): {preview}")

            # Build robot states: copy from library, prefer GPT plan list order if present.
            robots_plan = layout_plan.get("robots") or []
            robots_dicts: list[dict] = []
            if isinstance(robots_plan, list) and robots_plan:
                for r in robots_plan:
                    if not isinstance(r, dict):
                        continue
                    name = r.get("name")
                    if isinstance(name, str) and name in all_robots:
                        robots_dicts.append(
                            {
                                "name": name,
                                "pos": list(all_robots[name]["pos"]),
                                "rot": list(all_robots[name]["rot"]),
                                "dof_pos": dict(all_robots[name]["dof_pos"]),
                            }
                        )
            if not robots_dicts:
                for name in partial.get("robot_involved") or []:
                    if name in all_robots:
                        robots_dicts.append(
                            {
                                "name": name,
                                "pos": list(all_robots[name]["pos"]),
                                "rot": list(all_robots[name]["rot"]),
                                "dof_pos": dict(all_robots[name]["dof_pos"]),
                            }
                        )

            # Tabletop optimization (table is a special object category):
            # If the selected objects include a `table` object (and TASKGEN_ALLOW_TABLE=1),
            # then we place everything on the tabletop:
            # - constrain (x,y) to table.tabletop.range
            # - add table.tabletop.height to all objects' and robots' z (except the table itself)
            # - tell physics insertion to use the table object as the support surface
            allow_table = _env_flag("TASKGEN_ALLOW_TABLE", default=False)
            table_object_names: set[str] = set()
            if allow_table:
                entry = (self.obj_registry.get("categories") or {}).get("table") or {}
                names = entry.get("object_names") or []
                if isinstance(names, list):
                    table_object_names = {str(n) for n in names if isinstance(n, str)}
            table_name: str | None = None
            if allow_table:
                table_candidates = [n for n in target_set if n in table_object_names]
                if len(table_candidates) == 1:
                    table_name = table_candidates[0]

            tabletop_enabled = False
            tabletop_height = 0.0
            tabletop_range: dict[str, float] | None = None
            tabletop_z_world: float | None = None
            if table_name and table_name in all_objs:
                # Apply the table asset scale to tabletop metadata (height/range are authored at scale=1).
                scale_raw = all_objs[table_name].get("scale")
                if isinstance(scale_raw, list) and len(scale_raw) == 3 and all(isinstance(v, (int, float)) for v in scale_raw):
                    sx, sy, sz = float(scale_raw[0]), float(scale_raw[1]), float(scale_raw[2])
                else:
                    sx = sy = sz = 1.0

                tt = all_objs[table_name].get("tabletop")
                if isinstance(tt, dict) and isinstance(tt.get("height"), (int, float)) and isinstance(tt.get("range"), dict):
                    r = tt["range"]
                    if all(k in r for k in ("xmin", "xmax", "ymin", "ymax")):
                        tabletop_height = float(tt["height"]) * float(sz)
                        tabletop_range = {
                            "xmin": float(r["xmin"]) * float(sx),
                            "xmax": float(r["xmax"]) * float(sx),
                            "ymin": float(r["ymin"]) * float(sy),
                            "ymax": float(r["ymax"]) * float(sy),
                        }
                        tabletop_enabled = True
                if not tabletop_enabled:
                    # Fallback: estimate from MJCF (and treat result as range/height in the global tabletop frame).
                    mjcf = all_objs[table_name].get("mjcf_path")
                    if isinstance(mjcf, str) and mjcf:
                        est = LayoutValidator.estimate_table_asset_surface(mjcf_path=mjcf, margin_xy=0.03, top_band=0.02)
                    else:
                        est = None
                    if est:
                        tabletop_height = float(est["tabletop_z"]) * float(sz)
                        tabletop_range = {
                            "xmin": float(est["xmin"]) * float(sx),
                            "xmax": float(est["xmax"]) * float(sx),
                            "ymin": float(est["ymin"]) * float(sy),
                            "ymax": float(est["ymax"]) * float(sy),
                        }
                        tabletop_enabled = True
                        # Persist unscaled metadata; scaling is applied dynamically.
                        all_objs[table_name]["tabletop"] = {"height": float(est["tabletop_z"]), "range": {k: float(est[k]) for k in ("xmin","xmax","ymin","ymax")}}
                        print(f"  🪵 3.2t Table `{table_name}` tabletop inferred from MJCF.")
                    else:
                        # In allow_table mode, this is a hard error: we must have a valid tabletop height/range.
                        print(f"  ❌ 3.2t Table `{table_name}` selected but missing tabletop metadata; cannot place tabletop task.")
                        return None, {
                            "error_type": "tabletop_metadata_missing",
                            "table_object": table_name,
                            "hint": "Ensure table category assets have `tabletop.height` and `tabletop.range` in taskgen_json (or MJCF inference works).",
                        }

            if tabletop_enabled and tabletop_range:
                workspace["xmin"] = float(tabletop_range["xmin"])
                workspace["xmax"] = float(tabletop_range["xmax"])
                workspace["ymin"] = float(tabletop_range["ymin"])
                workspace["ymax"] = float(tabletop_range["ymax"])
                workspace["surface"] = "tabletop"
                workspace["support_object_name"] = str(table_name)
                # Store unscaled tabletop height; MuJoCo validator applies the asset Z scale consistently.
                if isinstance(tt, dict) and isinstance(tt.get("height"), (int, float)):
                    workspace["support_object_tabletop_height"] = float(tt["height"])
                else:
                    # MJCF-inferred heights are authored at scale=1.
                    workspace["support_object_tabletop_height"] = float(tabletop_height) / float(sz) if float(sz) != 0 else float(tabletop_height)
                print(
                    f"  🪵 3.2t Tabletop enabled via `{table_name}`: height≈{tabletop_height:.3f}, "
                    f"xy∈[{workspace['xmin']:.3f},{workspace['xmax']:.3f}]×[{workspace['ymin']:.3f},{workspace['ymax']:.3f}]"
                )

            # Finalize object poses deterministically with a greedy placement + fast repair.
            priorities: dict[str, int] = {}
            for name, p in placements.items():
                pr = p.get("priority")
                priorities[name] = int(_as_float(pr, 100 if name in involved_names else 10))
            for name in target_set:
                priorities.setdefault(name, 100 if name in involved_names else 10)

            order = sorted(list(target_set), key=lambda n: priorities.get(n, 0), reverse=True)
            if tabletop_enabled and table_name and table_name in order:
                order = [table_name] + [n for n in order if n != table_name]
            spiral = _spiral_offsets(step=0.03, turns=16, points_per_turn=18)

            def ok_spacing(x: float, y: float, placed: list[dict]) -> bool:
                for o in placed:
                    ox, oy, _ = o["pos"]
                    dx, dy = x - ox, y - oy
                    if (dx * dx + dy * dy) ** 0.5 < min_clearance:
                        return False
                return True

            placed_dicts: list[dict] = []
            for name in order:
                meta = placements.get(name, {"name": name, "xy": None, "yaw": 0.0})
                base_z = float(condensed_objs[name]["z"])
                # Do NOT use GPT rotations. Always use asset base rotation from the object library.
                base_q_raw = condensed_objs[name].get("rot") if isinstance(condensed_objs.get(name), dict) else None
                if isinstance(base_q_raw, list) and len(base_q_raw) == 4:
                    try:
                        base_q = _quat_norm([float(v) for v in base_q_raw])
                    except Exception:
                        base_q = [1.0, 0.0, 0.0, 0.0]
                else:
                    base_q = [1.0, 0.0, 0.0, 0.0]
                rot = base_q

                xy = _as_xy(meta.get("xy"))
                if xy is None:
                    x0 = random.uniform(float(workspace["xmin"]), float(workspace["xmax"]))
                    y0 = random.uniform(float(workspace["ymin"]), float(workspace["ymax"]))
                else:
                    x0, y0 = xy
                # If tabletop mode is enabled, keep the table centered at the origin.
                if tabletop_enabled and table_name and name == table_name:
                    x0, y0 = 0.0, 0.0

                chosen = None
                for dx, dy in spiral:
                    x = _clamp(float(x0) + dx, float(workspace["xmin"]), float(workspace["xmax"]))
                    y = _clamp(float(y0) + dy, float(workspace["ymin"]), float(workspace["ymax"]))
                    if ok_spacing(x, y, placed_dicts):
                        chosen = (x, y)
                        break
                if chosen is None:
                    for _ in range(200):
                        x = random.uniform(float(workspace["xmin"]), float(workspace["xmax"]))
                        y = random.uniform(float(workspace["ymin"]), float(workspace["ymax"]))
                        if ok_spacing(x, y, placed_dicts):
                            chosen = (x, y)
                            break
                if chosen is None:
                    chosen = (x0, y0)

                x, y = chosen
                if tabletop_enabled and table_name and name == table_name:
                    tabletop_z_world = float(base_z) + float(tabletop_height)
                    # Apply tabletop offset to robots too (requested).
                    for r in robots_dicts:
                        try:
                            rx, ry, rz = [float(v) for v in r["pos"]]
                        except Exception:
                            continue
                        r["pos"] = [
                            _clamp(rx, float(workspace["xmin"]), float(workspace["xmax"])),
                            _clamp(ry, float(workspace["ymin"]), float(workspace["ymax"])),
                            float(rz) + float(tabletop_z_world),
                        ]
                if tabletop_enabled and tabletop_z_world is not None and table_name and name != table_name:
                    base_z = float(base_z) + float(tabletop_z_world)
                placed_dicts.append(
                    {
                        "name": name,
                        "pos": [float(x), float(y), base_z],
                        "rot": [float(v) for v in rot],
                        "is_involved": name in involved_names,
                    }
                )

            mujoco_assets_subset = {o["name"]: all_objs[o["name"]] for o in placed_dicts if o["name"] in all_objs}

            def _physics_error_summary(rep: dict) -> str:
                e = (rep.get("errors") or [{}])[0]
                return f"physics:{rep.get('phase','?')}:{e.get('type','?')}:{e}"

            def _physics_repair_once(objs: list[dict], rep: dict, k: int) -> list[dict]:
                e = (rep.get("errors") or [{}])[0] if isinstance(rep, dict) else {}
                et = e.get("type")
                by_name = {o["name"]: dict(o) for o in objs}
                spiral2 = _spiral_offsets(step=0.02, turns=10, points_per_turn=18)

                def _raise(name: str, dz: float) -> None:
                    if name in by_name:
                        x, y, z = by_name[name]["pos"]
                        by_name[name]["pos"] = [float(x), float(y), float(z) + float(dz)]

                def _nudge(name: str) -> None:
                    if name in by_name:
                        x, y, z = by_name[name]["pos"]
                        dx, dy = spiral2[k % len(spiral2)]
                        by_name[name]["pos"] = [
                            _clamp(float(x) + dx, float(workspace["xmin"]), float(workspace["xmax"])),
                            _clamp(float(y) + dy, float(workspace["ymin"]), float(workspace["ymax"])),
                            float(z),
                        ]

                if et in {"support_penetration", "bad_height_or_support_penetration", "deep_penetration"}:
                    name = e.get("obj")
                    if not isinstance(name, str) and et == "deep_penetration":
                        # Fallback: infer object name from geom strings like "lamp//unnamed_geom_1".
                        g1 = e.get("geom1")
                        g2 = e.get("geom2")
                        kinds = e.get("kinds") or []
                        geom = None
                        if isinstance(kinds, list) and len(kinds) == 2:
                            if kinds[0] == "object" and isinstance(g1, str):
                                geom = g1
                            elif kinds[1] == "object" and isinstance(g2, str):
                                geom = g2
                        if geom is None and isinstance(g2, str):
                            geom = g2
                        if isinstance(geom, str) and "//" in geom:
                            name = geom.split("//", 1)[0]
                    dist = float(e.get("dist", 0.0))
                    kinds = e.get("kinds") or []
                    is_table_pen = isinstance(kinds, list) and ("table" in kinds and "object" in kinds)
                    is_ground_pen = isinstance(kinds, list) and ("ground" in kinds and "object" in kinds)

                    # For penetrations into supports, prefer gradual pose adjustment:
                    # - Table penetrations often need both Z lift and XY escape.
                    # - Ground penetrations are usually fixed by Z lift alone.
                    if isinstance(name, str) and name in by_name:
                        if is_table_pen:
                            # If penetration is huge, do a one-shot lift to exit overlap, then use smaller steps.
                            if dist < float(LayoutValidator.COLLISION_DIST):
                                dz = (-dist) + 0.01
                            else:
                                dz = 0.01
                            _raise(name, dz)
                            _nudge(name)
                        elif is_ground_pen:
                            dz = max(0.005, (-dist) + 0.005)
                            _raise(name, dz)
                        else:
                            # Scene/unknown support: lift conservatively + small XY nudge.
                            dz = max(0.01, min(0.05, (-dist) + 0.01))
                            _raise(name, dz)
                            _nudge(name)
                    return list(by_name.values())

                if et in {"collision", "collision_with_robot"}:
                    a = e.get("a") or e.get("obj")
                    b = e.get("b")
                    candidates = [x for x in [a, b] if isinstance(x, str) and x in by_name]
                    if candidates:
                        candidates.sort(key=lambda n: priorities.get(n, 0))
                        _nudge(candidates[0])
                    return list(by_name.values())

                if et in {"out_of_bounds", "below_ground"}:
                    name = e.get("obj")
                    if isinstance(name, str) and name in by_name:
                        x, y, z = by_name[name]["pos"]
                        by_name[name]["pos"] = [
                            _clamp(float(x), float(workspace["xmin"]), float(workspace["xmax"])),
                            _clamp(float(y), float(workspace["ymin"]), float(workspace["ymax"])),
                            max(float(z), 0.0),
                        ]
                    return list(by_name.values())

                decor_candidates = [n for n in by_name.keys() if n not in involved_names]
                if decor_candidates:
                    _nudge(decor_candidates[k % len(decor_candidates)])
                return list(by_name.values())

            print("  🛠️  3.3 Validating/repairing (poses only; fixed sets)...")
            repaired, report = self._layout_repairer.repair(
                objects=placed_dicts,
                robots=robots_dicts,
                workspace=workspace,
                min_clearance=min_clearance,
                priorities=priorities,
                validator=self._layout_validator,
                mujoco_assets=mujoco_assets_subset,
            )
            if not report.get("ok", False):
                errs = report.get("errors") or []
                print(
                    f"  ❌ 3.3a Fast validate failed after pose repair (mode={report.get('mode','?')}, iterations={report.get('iterations','?')}, errors={len(errs)})"
                )
                if errs:
                    print(f"  ❌ 3.3b Fast validate first error: {errs[0]}")
                failures.append(f"fast_validate_failed:{errs[:2]}")
                print("  🔁 3.3c Retrying GPT Stage3 due to fast validation failure...\n")
                continue

            print("  🧪 3.4 Physics verification (mj_forward + settle under gravity)...")
            max_validation_iters = 30
            for v_it in range(1, max_validation_iters + 1):
                physics_report = self._layout_validator.validate_with_incremental_insertion(
                    objects=repaired,
                    robots=robots_dicts,
                    workspace=workspace,
                    mujoco_assets=mujoco_assets_subset,
                    involved_objects=involved_list,
                    decor_objects=decor_list,
                    debug=True,
                    steps=10_000,
                    pen_tol=2e-4,
                    deep_pen=3e-3,
                    ground_pen_ok=2e-2,
                    per_obj_attempts=80,
                    # Stable-frame velocity threshold (m/s).
                    settle_stop_vel=1e-2,
                    settle_stop_frames=200,
                )
                if physics_report.get("ok", False):
                    break

                first_err = (physics_report.get("errors") or [{}])[0]
                print(f"  ⚠️  3.4b Incremental physics failed (iter={v_it}/{max_validation_iters}, phase={physics_report.get('phase','?')}): {first_err}")

                # Validation-stage fixes (never go back to Stage3 from here).
                if physics_report.get("mode") == "mujoco_incremental" and first_err.get("type") == "insertion_failed":
                    details = first_err.get("details") or {}
                    obj_new = first_err.get("obj")
                    print(f"  ⚠️  3.4c Insertion details: {details}")

                    # If the new object collides with another object, ask Stage4 LLM to adjust ONLY (x,y).
                    if isinstance(details, dict) and details.get("type") == "collision" and isinstance(obj_new, str):
                        print(f"  🤖 3.4d Stage4 LLM fix triggered for `{obj_new}` (collision with={details.get('with')})")
                        max_fix = 8
                        fixed = False
                        for fix_i in range(1, max_fix + 1):
                            by_name = {o["name"]: o for o in repaired}
                            cur = by_name.get(obj_new)
                            if not cur:
                                break
                            cur_xy = [float(cur["pos"][0]), float(cur["pos"][1])]
                            other_poses = [
                                {
                                    "name": o["name"],
                                    "pos": [float(o["pos"][0]), float(o["pos"][1])],
                                    "is_involved": bool(o.get("is_involved", False)),
                                }
                                for o in repaired
                                if o["name"] != obj_new
                            ]
                            ctx = {
                                "workspace": workspace,
                                "failed_object": {"name": obj_new, "current_pos": cur_xy},
                                "other_objects": other_poses,
                                "collision": {
                                    "with": details.get("with"),
                                    "dist": details.get("dist"),
                                },
                                "constraints": {
                                    "allowed_change": "(x,y) only",
                                    "keep_sets_fixed": True,
                                    "min_clearance_hint": min_clearance,
                                },
                            }
                            fix = self._call_gpt(format_stage4_prompt(ctx))
                            if not isinstance(fix, dict) or fix.get("name") != obj_new:
                                print(f"  ❌ 3.4d Stage4 fix attempt {fix_i}/{max_fix}: invalid response")
                                continue
                            pos2 = fix.get("pos")
                            if not isinstance(pos2, list) or len(pos2) != 2:
                                print(f"  ❌ 3.4d Stage4 fix attempt {fix_i}/{max_fix}: pos invalid")
                                continue
                            x2, y2 = float(pos2[0]), float(pos2[1])
                            if not (
                                float(workspace["xmin"]) <= x2 <= float(workspace["xmax"])
                                and float(workspace["ymin"]) <= y2 <= float(workspace["ymax"])
                            ):
                                print(f"  ❌ 3.4d Stage4 fix attempt {fix_i}/{max_fix}: out of bounds")
                                continue
                            old_xy = (float(cur["pos"][0]), float(cur["pos"][1]))
                            cur["pos"][0] = x2
                            cur["pos"][1] = y2
                            repaired = list(by_name.values())
                            print(
                                f"  🔧 3.4d Stage4 applied fix for {obj_new} (attempt {fix_i}/{max_fix}): xy=({old_xy[0]:.3f},{old_xy[1]:.3f})->({x2:.3f},{y2:.3f})"
                            )
                            fixed = True
                            break
                        if fixed:
                            continue

                    # For support penetrations/z-cap issues, do a deterministic re-sample of this object's XY (no GPT).
                    if isinstance(details, dict) and isinstance(obj_new, str) and details.get("type") in {"support_penetration", "z_cap_too_low"}:
                        by_name = {o["name"]: o for o in repaired}
                        cur = by_name.get(obj_new)
                        if cur is not None:
                            old_xy = (float(cur["pos"][0]), float(cur["pos"][1]))
                            x2 = random.uniform(float(workspace["xmin"]), float(workspace["xmax"]))
                            y2 = random.uniform(float(workspace["ymin"]), float(workspace["ymax"]))
                            cur["pos"][0] = x2
                            cur["pos"][1] = y2
                            repaired = list(by_name.values())
                            print(
                                f"  🧭 3.4d Support issue -> resampled `{obj_new}` xy=({old_xy[0]:.3f},{old_xy[1]:.3f})->({x2:.3f},{y2:.3f}) (no GPT)"
                            )
                            continue

                # Unhandled validation error types: stop and report failure (no Stage3 regeneration here).
                break

            if not physics_report.get("ok", False):
                failures.append(_physics_error_summary(physics_report))
                print("  ❌ 3.5 Validation failed after fixes (will not regenerate Stage3 in validation).")
                return None, {
                    "error_type": "validation_failed",
                    "repair_report": report,
                    "physics_report": physics_report,
                    "reasoning": layout_plan.get("reasoning", {}),
                }

            settled = physics_report.get("settled_objects") or {}
            if isinstance(settled, dict) and settled:
                by_name = {o["name"]: o for o in repaired}
                for name, pose in settled.items():
                    if name in by_name and isinstance(pose, dict):
                        if isinstance(pose.get("pos"), list) and len(pose["pos"]) == 3:
                            by_name[name]["pos"] = [float(v) for v in pose["pos"]]
                        if isinstance(pose.get("rot"), list) and len(pose["rot"]) == 4:
                            by_name[name]["rot"] = [float(v) for v in pose["rot"]]
                repaired = list(by_name.values())
                print(
                    f"  🧷 3.6 Using settled poses for {len(settled)} objects (steps_run={physics_report.get('steps_run','?')}, stable_frames={physics_report.get('stable_frames','?')}).\n"
                )
                if True:
                    # Print the exact poses that will be saved into task JSON/PKL (human-check loads these).
                    # This helps diagnose any mismatch between settle output and subsequent replay.
                    names = sorted(list(by_name.keys()))
                    print("  🧾 3.6a Saved object poses (post-settle, will be written):")
                    for n in names:
                        p = by_name[n].get("pos") or [0, 0, 0]
                        print(f"    - {n}: pos=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f})")

            settled_robots = physics_report.get("settled_robots") or {}
            if isinstance(settled_robots, dict) and settled_robots:
                by_rname = {r["name"]: r for r in robots_dicts if isinstance(r, dict) and isinstance(r.get("name"), str)}
                for name, pose in settled_robots.items():
                    if name in by_rname and isinstance(pose, dict):
                        if isinstance(pose.get("pos"), list) and len(pose["pos"]) == 3:
                            by_rname[name]["pos"] = [float(v) for v in pose["pos"]]
                        if isinstance(pose.get("rot"), list) and len(pose["rot"]) == 4:
                            by_rname[name]["rot"] = [float(v) for v in pose["rot"]]
                robots_dicts = list(by_rname.values())
                if True:
                    print("  🧾 3.6b Saved robot poses (post-adjust, will be written):")
                    for r in robots_dicts:
                        p = r.get("pos") or [0, 0, 0]
                        print(f"    - {r.get('name','?')}: pos=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f})")

            print(
                f"  ✅ 3.7 Layout OK (fast={report.get('mode','?')}, iters={report.get('iterations','?')}; physics={physics_report.get('mode','?')}, phase={physics_report.get('phase','?')})\n"
            )

            # Build TaskSpec from repaired layout.
            objects = [
                ObjectState(
                    name=o["name"],
                    pos=Position.from_list(o["pos"]),
                    rot=Rotation.from_list(o["rot"]),
                    usd_path=all_objs[o["name"]]["usd_path"],
                    urdf_path=all_objs[o["name"]].get("urdf_path"),
                    mjcf_path=all_objs[o["name"]].get("mjcf_path"),
                    scale=(
                        tuple(float(v) for v in all_objs[o["name"]].get("scale"))
                        if isinstance(all_objs[o["name"]].get("scale"), (list, tuple))
                        and len(all_objs[o["name"]].get("scale")) == 3
                        else None
                    ),
                    is_involved=bool(o.get("is_involved", False)),
                )
                for o in repaired
                if o["name"] in all_objs
            ]
            robots = [
                RobotState(
                    name=r["name"],
                    pos=Position.from_list(r["pos"]),
                    rot=Rotation.from_list(r["rot"]),
                    dof_pos=r.get("dof_pos", {}),
                )
                for r in robots_dicts
                if r["name"] in all_robots
            ]

            spec = TaskSpec(
                task_name=partial["task_name"],
                task_desc=partial["task_language_instruction"],
                objects=objects,
                robots=robots,
            )
            out = dict(layout_plan)
            out["workspace_final"] = workspace
            out["min_clearance_final"] = min_clearance
            out["repair_report"] = report
            out["physics_report"] = physics_report
            return spec, out

        return None, None

    def _load_objects(self, categories: list[str]) -> dict:
        result = {}
        for cat in categories:
            if cat not in self.obj_registry["categories"]:
                continue
            detail = self._load_json(Path(self.obj_registry["categories"][cat]["detail_file"]))
            for obj in detail.get("objects", []):
                paths = obj.get("paths", {})
                result[obj["name"]] = {
                    "usd_path": paths.get("usd_path"),
                    "urdf_path": paths.get("urdf_path"),
                    "mjcf_path": paths.get("mjcf_path"),
                    "init_state": {"pos": obj["init_state"]["pos"], "rot": obj["init_state"]["rot"]},
                    "tabletop": obj.get("tabletop"),
                    "scale": obj.get("scale") or [1.0, 1.0, 1.0],
                }
        return result

    def _load_robots(self, categories: list[str]) -> dict:
        result = {}
        for cat in categories:
            if cat not in self.robot_registry["categories"]:
                continue
            detail = self._load_json(Path(self.robot_registry["categories"][cat]["detail_file"]))
            for robot in detail.get("robots", []):
                result[robot["name"]] = {
                    "pos": robot["init_state"]["pos"],
                    "rot": robot["init_state"]["rot"],
                    "dof_pos": robot["init_state"]["dof_pos"],
                }
        return result


# ======================================
# Code Generation
# ======================================


class TaskFileTemplate:
    """Template for generating task Python files."""

    TEMPLATE = '''"""GPT-generated task: {task_desc}"""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .gpt_base import GptBaseTask


@register_task("gpt.{snake_name}", "gpt:{camel_name}")
class {class_name}(GptBaseTask):
    scenario = ScenarioCfg(
        objects=[
{objects_block}
        ],
        robots=["franka"],
    )
    max_episode_steps = 250
    task_desc = "{task_desc}"
    traj_filepath = "{traj_path}"
'''

    @classmethod
    def render(cls, spec: TaskSpec, paths: PathConfig) -> str:
        objects_block = cls._render_objects(spec.objects)
        traj_path = paths.pkl_output / spec.snake_name / "franka_v2.pkl"
        return cls.TEMPLATE.format(
            task_desc=spec.task_desc,
            snake_name=spec.snake_name,
            camel_name=spec.camel_name,
            class_name=spec.class_name,
            objects_block=objects_block,
            traj_path=traj_path,
        )

    @staticmethod
    def _render_objects(objects: list[ObjectState]) -> str:
        lines = []
        for obj in objects:
            parts = [f'name="{obj.name}"', "physics=PhysicStateType.RIGIDBODY"]
            if getattr(obj, "scale", None) is not None:
                sx, sy, sz = obj.scale
                parts.append(f"scale=({float(sx)}, {float(sy)}, {float(sz)})")
            if obj.usd_path:
                parts.append(f'usd_path="{obj.usd_path}"')
            if obj.urdf_path:
                parts.append(f'urdf_path="{obj.urdf_path}"')
            if obj.mjcf_path:
                parts.append(f'mjcf_path="{obj.mjcf_path}"')
            line = f'            RigidObjCfg({", ".join(parts)}),'
            lines.append(line)
        return "\n".join(lines)


# ======================================
# File Writers
# ======================================


class TaskWriter:
    """Writes task artifacts to filesystem."""

    def __init__(self, paths: PathConfig):
        self._paths = paths

    def write_all(self, spec: TaskSpec, reasoning_metadata: dict = None) -> tuple[str, str, str]:
        """Write JSON, PKL, and Python file. Returns paths."""
        json_path = self._write_json(spec, reasoning_metadata)
        pkl_path = self._write_pkl(spec)
        py_path = self._write_python(spec)
        return json_path, pkl_path, py_path

    def _write_json(self, spec: TaskSpec, reasoning_metadata: dict = None) -> str:
        path = self._paths.tasks_output / f"{spec.snake_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task_name": spec.task_name,
            "task_language_instruction": spec.task_desc,
            "objects_involved": [o.name for o in spec.objects if o.is_involved],
            "objects": [
                {
                    "name": o.name,
                    "pos": o.pos.to_list(),
                    "rot": o.rot.to_list(),
                    **({"scale": list(o.scale)} if getattr(o, "scale", None) is not None else {}),
                }
                for o in spec.objects
            ],
            "robots": [
                {"name": r.name, "pos": r.pos.to_list(), "rot": r.rot.to_list(), "dof_pos": r.dof_pos}
                for r in spec.robots
            ],
        }
        if reasoning_metadata:
            data["generation_metadata"] = reasoning_metadata
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    def _write_pkl(self, spec: TaskSpec) -> str:
        folder = self._paths.pkl_output / spec.snake_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "franka_v2.pkl"

        # Build legacy init_state format for compatibility
        init_state = {}
        for obj in spec.objects:
            init_state[obj.name] = {
                "pos": obj.pos.to_list(),
                "rot": obj.rot.to_list(),
                **({"scale": list(obj.scale)} if getattr(obj, "scale", None) is not None else {}),
            }
        for robot in spec.robots:
            init_state[robot.name] = {"pos": robot.pos.to_list(), "rot": robot.rot.to_list(), "dof_pos": robot.dof_pos}

        # Build trajectory data
        traj_data = {}
        for robot in spec.robots:
            zero_dof = {k: 0.0 for k in robot.dof_pos}
            traj_data[robot.name] = [
                {"actions": [{"dof_pos_target": zero_dof}], "init_state": init_state, "states": [], "extra": None}
            ]

        with open(path, "wb") as f:
            pickle.dump(traj_data, f)
        return str(path)

    def _write_python(self, spec: TaskSpec) -> str:
        path = self._paths.task_py_output / f"{spec.snake_name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = TaskFileTemplate.render(spec, self._paths)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)


# ======================================
# UI
# ======================================


class TaskGenUI:
    """Centralized UI for task generation."""

    @staticmethod
    def get_user_prompt() -> str:
        print(Fore.YELLOW + emoji.emojize("🔥 What can I help you with today? ✨") + Style.RESET_ALL)
        prompt = input("> ").strip()
        return prompt or "Please generate an interesting task for me."

    @staticmethod
    def get_num_tasks() -> int:
        print(Fore.YELLOW + emoji.emojize("🔢 How many tasks to generate? (default: 1)") + Style.RESET_ALL)
        try:
            n = input("> ").strip()
            return int(n) if n else 1
        except ValueError:
            return 1

    @staticmethod
    def get_difficulty() -> int:
        print(Fore.YELLOW + emoji.emojize("⚡ Select difficulty (1-5, default: 3):") + Style.RESET_ALL)
        for level, info in DIFFICULTY_DEFINITIONS.items():
            print(f"  {level}. {info['name']}: {info['objects_involved']} involved, {info['objects_decorative']} decorative")
        try:
            d = input("> ").strip()
            level = int(d) if d else 3
            return max(1, min(5, level))
        except ValueError:
            return 3

    @staticmethod
    def print_stage_start(stage: int, message: str) -> None:
        print(Fore.CYAN + emoji.emojize(f"{message}") + Style.RESET_ALL)

    @staticmethod
    def print_error(message: str) -> None:
        print(Fore.RED + f"  ❌ {message}" + Style.RESET_ALL)

    @staticmethod
    def print_layout(spec: TaskSpec) -> None:
        """Print ASCII visualization of layout (top-down view, x=right, y=up)."""
        W, H, bounds = 25, 13, 0.6
        grid = [[" "] * W for _ in range(H)]
        to_col = lambda x: int((x + bounds) / (2 * bounds) * (W - 1))
        to_row = lambda y: H - 1 - int((y + bounds) / (2 * bounds) * (H - 1))

        for r in spec.robots:
            c, r_ = to_col(r.pos.x), to_row(r.pos.y)
            if 0 <= c < W and 0 <= r_ < H:
                grid[r_][c] = "R"
        for o in spec.objects:
            c, r_ = to_col(o.pos.x), to_row(o.pos.y)
            if 0 <= c < W and 0 <= r_ < H:
                grid[r_][c] = "+" if o.is_involved else "×"

        print(Fore.CYAN + "  📍 Layout (top-down):" + Style.RESET_ALL)
        print("    +" + "-" * W + "+")
        for row in grid:
            line = "".join(
                Fore.GREEN + c + Style.RESET_ALL if c == "R" else
                Fore.YELLOW + c + Style.RESET_ALL if c == "+" else
                Style.DIM + c + Style.RESET_ALL if c == "×" else c
                for c in row
            )
            print(f"    |{line}|")
        print("    +" + "-" * W + "+")
        print(f"    {Fore.GREEN}R{Style.RESET_ALL}=robot  {Fore.YELLOW}+{Style.RESET_ALL}=involved  {Style.DIM}×{Style.RESET_ALL}=decorative\n")

    @staticmethod
    def print_summary(spec: TaskSpec, json_path: str, pkl_path: str, py_path: str) -> None:
        print("\n" + Fore.GREEN + emoji.emojize("🚀 The task has been generated! 🎉") + Style.RESET_ALL)
        print(Fore.CYAN + "🔹 Task Name: " + Style.BRIGHT + spec.task_name + Style.RESET_ALL)
        print(Fore.MAGENTA + "📝 Instruction: " + Style.BRIGHT + spec.task_desc + Style.RESET_ALL + "\n")
        print(Fore.BLUE + "📁 Files saved:" + Style.RESET_ALL)
        print(Fore.YELLOW + f"1. 🗂️ JSON: {json_path}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"2. 📦 PKL:  {pkl_path}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"3. 🐍 PY:   {py_path}" + Style.RESET_ALL + "\n")
        print(Fore.GREEN + emoji.emojize("🎮 Replay with:") + Style.RESET_ALL)
        print(Fore.WHITE + f"  python scripts/advanced/replay_demo.py --sim=mujoco --task=gpt.{spec.snake_name} --num_envs 1 --headless" + Style.RESET_ALL + "\n")

    @staticmethod
    def print_task_progress(current: int, total: int) -> None:
        print(Fore.GREEN + emoji.emojize(f"\n{'='*50}\n🎲 Task {current}/{total}\n{'='*50}") + Style.RESET_ALL)

    @staticmethod
    def print_final_summary(success_count: int, total: int) -> None:
        print(Fore.GREEN + emoji.emojize(f"🎉 Generated {success_count}/{total} tasks!") + Style.RESET_ALL)


# ======================================
# Main
# ======================================


def main():
    gpt_cfg = GPTConfig.from_env()
    if not gpt_cfg.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = openai.OpenAI(api_key=gpt_cfg.api_key, base_url=gpt_cfg.base_url)
    paths = PathConfig()
    generator = TaskGenerator(client, paths)
    writer = TaskWriter(paths)

    user_prompt = TaskGenUI.get_user_prompt()
    num_tasks = TaskGenUI.get_num_tasks()
    difficulty = TaskGenUI.get_difficulty()
    success_count = 0

    for i in range(num_tasks):
        if num_tasks > 1:
            TaskGenUI.print_task_progress(i + 1, num_tasks)

        spec, reasoning_metadata = generator.generate(user_prompt, difficulty)
        if not spec:
            continue
        json_path, pkl_path, py_path = writer.write_all(spec, reasoning_metadata)
        TaskGenUI.print_summary(spec, json_path, pkl_path, py_path)
        success_count += 1

    if num_tasks > 1:
        TaskGenUI.print_final_summary(success_count, num_tasks)


if __name__ == "__main__":
    with _filter_stderr_prefixes():
        main()
