# AutoTaskGeneration

This repository documents and contains helper scripts for an **LLM-driven robotic task generation pipeline** in a simulation-based robotics stack, plus utilities for **asset cataloging** and **human-in-the-loop pose/layout adjustment**.

## Workflow

1. **Catalog assets** into `taskgen_json/`.
2. **Generate tasks** (LLM stage 1–3: select → specify → layout/validate).
3. **Optional**: refine asset pose/scale and task layouts, or delete generated artifacts.

## Files

- `gpt_gen.py`: Generates GPT-authored tasks via a 3-stage LLM pipeline and writes JSON/PKL/PY artifacts.
- `gpt_layout_adjust.py`: Applies manually saved layout poses to update a task’s JSON and PKL `init_state`.
- `taskgen/manage_asset.py`: Classifies assets and writes per-category detail JSON.
- `taskgen/physical_pose_adjust.py`: Lets a human adjust an asset’s pose/scale in MuJoCo.
- `taskgen/manage_gpt_task.py`: Provides an interactive TUI to manage generated task artifacts.
