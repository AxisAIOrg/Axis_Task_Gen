#!/usr/bin/env python3
from __future__ import annotations

import curses
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger as log


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Args:
    tasks_json_dir: str = "metasim/cfg/tasks/gpt/config/tasks"
    """Directory containing generated task JSON files (task_snake.json)."""

    traj_dir: str = "roboverse_data/trajs/gpt"
    """Directory containing generated trajectories (task_snake/franka_v2.pkl)."""

    task_py_dir: str = "roboverse_pack/tasks/gpt"
    """Directory containing generated python tasks (task_snake.py)."""

    query: str | None = None
    """Substring filter for task names (case-insensitive)."""

    dry_run: bool = False
    yes: bool = False
    """Skip the final confirmation prompt."""


@dataclass(frozen=True)
class TaskEntry:
    snake: str
    json_path: Path
    pkl_dir: Path
    py_path: Path
    pycache_paths: tuple[Path, ...]


def _is_tty() -> bool:
    return sys.stdin is not None and sys.stdout is not None and sys.stdin.isatty() and sys.stdout.isatty()


def _discover_tasks(args: Args) -> list[TaskEntry]:
    tasks_dir = (_REPO_ROOT / args.tasks_json_dir).resolve()
    traj_dir = (_REPO_ROOT / args.traj_dir).resolve()
    py_dir = (_REPO_ROOT / args.task_py_dir).resolve()

    if not tasks_dir.exists():
        raise FileNotFoundError(tasks_dir)
    tasks = []
    q = (args.query or "").strip().lower()
    for p in sorted(tasks_dir.glob("*.json")):
        snake = p.stem
        if not snake:
            continue
        if q and q not in snake.lower():
            continue
        json_path = p
        pkl_dir = traj_dir / snake
        py_path = py_dir / f"{snake}.py"
        pycache = py_dir / "__pycache__"
        pycache_paths: list[Path] = []
        if pycache.exists():
            pycache_paths.extend(sorted(pycache.glob(f"{snake}.*.pyc")))
        tasks.append(
            TaskEntry(
                snake=snake,
                json_path=json_path,
                pkl_dir=pkl_dir,
                py_path=py_path,
                pycache_paths=tuple(pycache_paths),
            )
        )
    return tasks


def _render_line(entry: TaskEntry, selected: bool, width: int) -> str:
    mark = "[x]" if selected else "[ ]"
    parts = [mark, entry.snake]
    tags = []
    if entry.json_path.exists():
        tags.append("json")
    if entry.py_path.exists():
        tags.append("py")
    if entry.pkl_dir.exists():
        tags.append("pkl")
    if entry.pycache_paths:
        tags.append("pyc")
    suffix = f" ({','.join(tags)})" if tags else ""
    s = " ".join(parts) + suffix
    if len(s) > width - 1:
        s = s[: max(0, width - 2)] + "…"
    return s


def _task_entry_has_any(entry: TaskEntry) -> bool:
    return entry.json_path.exists() or entry.py_path.exists() or entry.pkl_dir.exists() or bool(entry.pycache_paths)


def _delete_entry(entry: TaskEntry, *, dry_run: bool) -> None:
    targets: list[Path] = []
    if entry.json_path.exists():
        targets.append(entry.json_path)
    if entry.py_path.exists():
        targets.append(entry.py_path)
    if entry.pkl_dir.exists():
        targets.append(entry.pkl_dir)
    targets.extend([p for p in entry.pycache_paths if p.exists()])

    for t in targets:
        if dry_run:
            log.info("[dry-run] delete {}", t)
            continue
        try:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            log.info("deleted {}", t)
        except Exception as e:
            log.warning("failed to delete {}: {}", t, e)


def _select_tasks_curses(entries: list[TaskEntry]) -> list[TaskEntry]:
    if not entries:
        return []

    selected = [False] * len(entries)

    def _run(stdscr) -> list[TaskEntry]:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        idx = 0
        top = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            header = "manage_gpt_task: ↑/↓ move  SPACE toggle  a all  n none  ENTER delete  q quit"
            stdscr.addnstr(0, 0, header, w - 1)

            visible_h = max(1, h - 2)
            if idx < top:
                top = idx
            if idx >= top + visible_h:
                top = idx - visible_h + 1

            for row in range(visible_h):
                i = top + row
                if i >= len(entries):
                    break
                line = _render_line(entries[i], selected[i], w)
                if i == idx:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addnstr(1 + row, 0, line, w - 1)
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addnstr(1 + row, 0, line, w - 1)

            footer = f"selected: {sum(1 for x in selected if x)} / {len(entries)}"
            stdscr.addnstr(h - 1, 0, footer, w - 1)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q"), 27):  # q or ESC
                return []
            if ch in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
                continue
            if ch in (curses.KEY_DOWN, ord("j")):
                idx = min(len(entries) - 1, idx + 1)
                continue
            if ch == ord(" "):
                selected[idx] = not selected[idx]
                continue
            if ch in (ord("a"), ord("A")):
                for i in range(len(selected)):
                    selected[i] = True
                continue
            if ch in (ord("n"), ord("N")):
                for i in range(len(selected)):
                    selected[i] = False
                continue
            if ch in (curses.KEY_ENTER, 10, 13):
                return [entries[i] for i, ok in enumerate(selected) if ok]

    return curses.wrapper(_run)


def _confirm_delete(entries: list[TaskEntry], *, yes: bool) -> bool:
    if yes:
        return True
    print("\nSelected tasks to delete:")
    for e in entries[:30]:
        print(f"- {e.snake}")
    if len(entries) > 30:
        print(f"... and {len(entries) - 30} more")
    print("\nType DELETE to confirm:")
    try:
        ans = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans == "DELETE"


def main() -> None:
    args = tyro.cli(Args)
    entries = _discover_tasks(args)
    if not entries:
        print("No GPT tasks found.")
        return

    if not _is_tty():
        raise RuntimeError("Interactive selection requires a TTY. Run in a terminal, or add a non-interactive mode.")

    chosen = _select_tasks_curses(entries)
    if not chosen:
        print("No tasks selected.")
        return

    # Filter out entries that no longer exist (paranoia).
    chosen = [e for e in chosen if _task_entry_has_any(e)]
    if not chosen:
        print("Selected tasks have no artifacts on disk.")
        return

    if not _confirm_delete(chosen, yes=args.yes):
        print("Cancelled.")
        return

    for e in chosen:
        _delete_entry(e, dry_run=args.dry_run)

    print(f"Done. deleted={len(chosen)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
