from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.action_queue import empty_action_document, save_action_document
from runtime.environment_io import default_environment_doc, save_environment_document


DEFAULT_WORKSPACE = Path("workspace")


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    action: Path
    environment: Path
    embodied: Path
    lessons: Path
    task: Path
    skill: Path
    plan: Path
    report: Path


def workspace_paths(root: str | Path = DEFAULT_WORKSPACE) -> WorkspacePaths:
    path = Path(root)
    return WorkspacePaths(
        root=path,
        action=path / "ACTION.md",
        environment=path / "ENVIRONMENT.md",
        embodied=path / "EMBODIED.md",
        lessons=path / "LESSONS.md",
        task=path / "TASK.md",
        skill=path / "SKILL.md",
        plan=path / "PLAN.md",
        report=path / "REPORT.md",
    )


def initialize_workspace(root: str | Path = DEFAULT_WORKSPACE, *, overwrite: bool = False) -> WorkspacePaths:
    paths = workspace_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)

    if overwrite or not paths.action.exists():
        save_action_document(paths.action, empty_action_document())
    if overwrite or not paths.environment.exists():
        save_environment_document(paths.environment, default_environment_doc())
    if overwrite or not paths.embodied.exists():
        paths.embodied.write_text(default_embodied_profile(), encoding="utf-8")
    if overwrite or not paths.lessons.exists():
        paths.lessons.write_text("# Lessons\n\nNo lessons recorded yet.\n", encoding="utf-8")
    if overwrite or not paths.task.exists():
        paths.task.write_text("# Task\n\n- status: idle\n- instruction: none\n", encoding="utf-8")
    if overwrite or not paths.skill.exists():
        paths.skill.write_text("# Skill\n\nNo reusable workflow recorded yet.\n", encoding="utf-8")
    if overwrite or not paths.plan.exists():
        paths.plan.write_text("# Task Plan\n\nNo plan generated yet.\n", encoding="utf-8")
    if overwrite or not paths.report.exists():
        paths.report.write_text("# Execution Report\n\nNo execution report generated yet.\n", encoding="utf-8")

    return paths


def default_embodied_profile() -> str:
    return """# EMBODIED.md

## Identity

- Name: FakeManipulationEnv
- Type: 2D language-conditioned pick-and-place environment
- Driver: fake_manipulation

## Sensors

- RGB top-down render: optional 128x128 image
- State observation: end-effector position, gripper state, object positions, receptacle positions

## Supported Actions

| Action | Parameters | Description |
| --- | --- | --- |
| `reset` | `instruction`, `target_color`, `receptacle_name` | Reset the fake manipulation task. |
| `env_step` | `action: [dx, dy, gripper]` | Step the environment once. `gripper > 0` closes, `gripper <= 0` opens. |

## Physical Constraints

- Workspace bounds: x/y in `[-1.0, 1.0]`
- Max xy delta per step: `0.06`
- Grasp radius: `0.08`
- Place radius: `0.10`
- Objects: red, blue, and green blocks
- Receptacle: bowl

## Runtime Protocol

- Pending actions are read from `ACTION.md`.
- Runtime state is written to `ENVIRONMENT.md`.
- Failed or unsafe action patterns should be recorded in `LESSONS.md`.
"""
