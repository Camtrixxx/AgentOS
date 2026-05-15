from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.action_queue import (
    empty_action_document,
    load_action_document,
    save_action_document,
)
from runtime.environment_io import (
    default_environment_doc,
    load_environment_document,
    save_environment_document,
)
from runtime.lessons import append_lesson as _append_lesson_raw
from runtime.plan_io import save_execution_report, save_plan_document
from runtime.planner import TaskPlan
from runtime.workspace import (
    WorkspacePaths,
    default_embodied_profile,
    workspace_paths,
)


def resolve_repo(
    default: str | Path | WorkspacePaths | None,
    parameters: dict[str, Any] | None = None,
) -> "WorkspaceRepository":
    """Resolve a workspace specification to a WorkspaceRepository.

    Priority: parameters['workspace'] > default > 'workspace'.
    """
    ws = parameters.get("workspace") if parameters is not None else None
    if ws is not None:
        return WorkspaceRepository(Path(ws))
    if isinstance(default, WorkspaceRepository):
        return default
    if default is not None:
        return WorkspaceRepository(default)
    return WorkspaceRepository()


class WorkspaceRepository:
    """File-backed workspace abstraction.

    Wraps all Markdown file I/O so callers operate through domain methods
    instead of direct file paths — enabling future swap to Redis, NATS, or gRPC.
    """

    def __init__(self, root: str | Path | WorkspacePaths = "workspace") -> None:
        if isinstance(root, WorkspacePaths):
            self._paths = root
        else:
            self._paths = workspace_paths(Path(root))

    @property
    def paths(self) -> WorkspacePaths:
        return self._paths

    # -- initialization --

    def initialize(self, *, overwrite: bool = False) -> None:
        p = self._paths
        p.root.mkdir(parents=True, exist_ok=True)

        if overwrite or not p.action.exists():
            save_action_document(p.action, empty_action_document())
        if overwrite or not p.environment.exists():
            save_environment_document(p.environment, default_environment_doc())
        if overwrite or not p.embodied.exists():
            p.embodied.write_text(default_embodied_profile(), encoding="utf-8")
        if overwrite or not p.lessons.exists():
            p.lessons.write_text("# Lessons\n\nNo lessons recorded yet.\n", encoding="utf-8")
        if overwrite or not p.task.exists():
            p.task.write_text("# Task\n\n- status: idle\n- instruction: none\n", encoding="utf-8")
        if overwrite or not p.skill.exists():
            p.skill.write_text("# Skill\n\nNo reusable workflow recorded yet.\n", encoding="utf-8")
        if overwrite or not p.plan.exists():
            p.plan.write_text("# Task Plan\n\nNo plan generated yet.\n", encoding="utf-8")
        if overwrite or not p.report.exists():
            p.report.write_text("# Execution Report\n\nNo execution report generated yet.\n", encoding="utf-8")

    # -- environment --

    def get_environment(self) -> dict[str, Any]:
        return load_environment_document(self._paths.environment)

    def save_environment(self, document: dict[str, Any]) -> None:
        save_environment_document(self._paths.environment, document)

    # -- actions --

    def get_actions(self) -> dict[str, Any]:
        return load_action_document(self._paths.action)

    def save_actions(self, document: dict[str, Any]) -> None:
        save_action_document(self._paths.action, document)

    # -- plan / report --

    def save_plan(self, plan: TaskPlan) -> None:
        save_plan_document(self._paths.plan, plan)

    def save_report(self, report: dict[str, Any]) -> None:
        save_execution_report(self._paths.report, report)

    # -- lessons --

    def append_lesson(self, *, title: str, details: str) -> None:
        _append_lesson_raw(self._paths.lessons, title=title, details=details)
