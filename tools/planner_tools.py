from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.plan_io import plan_to_dict
from runtime.planner import RuleBasedPlanner
from runtime.repository import WorkspaceRepository, resolve_repo
from tools.response import ToolResponse


class CreatePlanTool:
    name = "create_plan"
    description = "Create and save a rule-based task plan for the fake manipulation domain."

    def __init__(self, workspace: str | Path | WorkspaceRepository = "workspace"):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.planner = RuleBasedPlanner()

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        instruction = str(parameters.get("instruction") or "pick up the red block and place it in the bowl")
        target_color = parameters.get("target_color")
        plan = self.planner.plan(instruction, target_color=str(target_color) if target_color else None)
        repo.initialize()
        repo.save_plan(plan)
        return ToolResponse.success(
            "plan created",
            data={"plan": plan_to_dict(plan), "path": str(repo.paths.plan)},
        )
