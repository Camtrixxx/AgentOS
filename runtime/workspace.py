from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    from runtime.repository import WorkspaceRepository

    repo = WorkspaceRepository(root)
    repo.initialize(overwrite=overwrite)
    return repo.paths


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
