<div align="center">

# AxisTaskGen

An end-to-end **LLM-powered task generation pipeline** for robotic manipulation  
**assets → registries → GPT task spec → HumanCheck layout → outputs**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-LLM-green)](https://openai.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Quick Start](#quick-start) • [Docs](#table-of-contents) • [Core Scripts](#core-scripts) • [File Structure](#file-structure)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model API / Keys](#model-api--keys)
- [Assets](#assets)
- [Core Scripts](#core-scripts)
  - [Generate tasks](#generate-tasks)
  - [Generate + HumanCheck](#generate--humancheck)
  - [Manage assets](#manage-assets)
  - [Adjust asset pose](#adjust-asset-pose)
  - [Manage generated tasks](#manage-generated-tasks)
- [Outputs](#outputs)
- [File Structure](#file-structure)
- [Notes / Dependencies](#notes--dependencies)

---

## Overview

TaskGen is designed to:
- Manage and register assets (rigid + articulated) into `taskgen_json/` registries.
- Generate manipulation tasks in **JSON + PKL + Python task class**, optionally **human-check** layouts in a UI.
- Support multiple LLM providers (OpenAI, DeepSeek, custom OpenAI-compatible endpoints).

All model-provider configuration is centralized in `taskgen/model_api.py`.

### Demo

<video width="100%" controls>
  <source src="assets/demo/taskgen_pipeline.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Core Pipeline

1) **Prepare assets** - Register assets into `taskgen_json/` and adjust initial poses if needed.
2) **Generate tasks** - Use `gpt_gen.py` to create a task spec and initial layout.
3) **HumanCheck (optional)** - Use `gpt_layout_adjust.py` to manually adjust object positions.
4) **Integrate & Use** - Use the generated task files in your simulation environment.

---

## Quick Start

### 1) Clone
```bash
git clone <repo-url>
cd Axis_Task_Gen
```

### 2) Configure model provider (env recommended)

TaskGen uses the OpenAI Python SDK for chat-completions, but the provider is selected via
`taskgen/model_api.py` and/or environment variables.

**Generic override**, works for any OpenAI-compatible base URL:

```bash
export MODEL_API_KEY="..."
export MODEL_BASE_URL="https://api.openai.com/v1"   # or yourV custom endpoint
export MODEL_MODEL="gpt-4o-2024-08-06"              # or "deepseek-chat"
```

Optional tuning:

```bash
export MODEL_TEMPERATURE=0.3
export MODEL_MAX_TOKENS=8192
export MODEL_SLEEP_S=0.0
```

### 3) Generate a task

```bash
python gpt_gen.py
```

### 4) (Optional) Generate + HumanCheck layout

```bash
python gpt_layout_adjust.py --prompt "Put the ketchup into the basket" --human-check
```

### 5) Use generated outputs

The generated JSON, PKL, and Python files can be integrated into your simulation environment.

---

## Model API / Keys

All provider logic is centralized in `taskgen/model_api.py`

You can configure TaskGen in two ways.

### Option A: Edit one file (global default)

Edit `taskgen/model_api.py` and set:

```py
ACTIVE_PROFILE = "openai"  # or "deepseek"
```

### Option B: Environment variables (recommended for CLI)

**Generic override** (works for any compatible base URL):

```bash
export MODEL_API_KEY="..."
export MODEL_BASE_URL="https://api.openai.com/v1"   # or your custom endpoint
export MODEL_MODEL="gpt-4o-2024-08-06"              # or "deepseek-chat"
```

**Provider-specific variables**:

```bash
# OpenAI
export MODEL_PROFILE=openai
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-2024-08-06"

# DeepSeek
export MODEL_PROFILE=deepseek
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/"
export DEEPSEEK_MODEL="deepseek-chat"
```

---

## Assets

### Asset folder format

TaskGen expects each asset to follow this layout:

```text
<asset_name>/
  mesh/                 # meshes + materials + textures
  mjcf/model.xml        # MJCF (MuJoCo)
  urdf/model.urdf       # optional
  usd/model.usd         # optional
  description.txt       # optional
```

### Where assets live

TaskGen-managed assets are registered in `taskgen_json/` directory with references to:

* Rigid assets
* Articulated assets

---

## Core Scripts

### Generate tasks

Use `gpt_gen.py` for automatic task generation:

```bash
python gpt_gen.py
```

This will:
* Interactively select environment and asset categories
* Generate task specification via LLM
* Create initial object layout
* Output: JSON task config, PKL trajectory init state, Python task class

---

### Generate + HumanCheck

Use `gpt_layout_adjust.py` to generate a task and optionally launch the manual layout tool:

```bash
python gpt_layout_adjust.py --prompt "Put the ketchup into the basket" --human-check
```

This allows you to:
* Generate a task from a natural language prompt
* Open an interactive UI to manually adjust object positions
* Save the corrected layout back to JSON/PKL

---

### Manage assets

Use `taskgen/manage_asset.py` to classify assets into categories and register them into `taskgen_json/` registries:

```bash
python taskgen/manage_asset.py <path_to_assets> --object-type rigid
```

Options:
* `--object-type`: Choose `rigid` or `articulated`
* The script uses LLM to automatically categorize assets

---

### Adjust asset pose

Use `taskgen/physical_pose_adjust.py` when an asset loads with incorrect initial position/rotation/scale:

```bash
python taskgen/physical_pose_adjust.py
```

This opens an interactive viewer and writes the corrected init state back into `taskgen_json/` detail JSON.

Tip: Run `manage_asset.py` first to register assets.

---

### Manage generated tasks

Use `taskgen/manage_gpt_task.py` for interactive cleanup of generated tasks:

```bash
python taskgen/manage_gpt_task.py
```

This provides a TUI (Text User Interface) to:
* Browse generated tasks
* Delete unwanted tasks (JSON/PKL/PY/pyc files)
* Clean up the task registry

---

## Outputs

Generated task files are placed in configurable output directories:

* **Task JSON config**: Contains task specification, object list, and initial poses
* **Task PKL (trajectory init state)**: Serialized initial state for simulation
* **Task Python class**: Executable task class for integration

---

## File Structure

```
Axis_Task_Gen/
├── gpt_gen.py                    # Main task generator (interactive)
├── gpt_layout_adjust.py          # Task generator + HumanCheck UI
├── taskgen/
│   ├── __init__.py
│   ├── manage_asset.py           # Asset classification & registration
│   ├── manage_gpt_task.py        # Task cleanup TUI
│   ├── physical_pose_adjust.py   # Asset pose adjustment tool
│   └── model_api.py              # LLM provider configuration (not shown but referenced)
└── taskgen_json/                 # Asset registries (created at runtime)
```

---

## Notes / Dependencies

* Common Python deps: `openai`, `trimesh`, and others depending on your simulation backend
* For asset processing, additional tools may be needed (e.g., `urdf2mjcf` for URDF conversion)
* If you see auth/provider issues, check:
  * `taskgen/model_api.py`
  * Your env vars (`MODEL_*`, `OPENAI_*`, `DEEPSEEK_*`)

---

## 🔗 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

<div align="center">

[Back to Top](#taskgen)

---

Built for robotic task generation and AI-powered automation

</div>
