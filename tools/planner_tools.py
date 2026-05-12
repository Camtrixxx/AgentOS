from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.plan_io import plan_to_dict, save_plan_document
from runtime.planner import RuleBasedPlanner
from runtime.workspace import initialize_workspace
from tools.response import ToolResponse


class CreatePlanTool:
    name = "create_plan"
    description = "Create and save a rule-based task plan for the fake manipulation domain."

    def __init__(self, workspace: str | Path = "workspace"):
        self.workspace = Path(workspace)
        self.planner = RuleBasedPlanner()

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        workspace = Path(parameters.get("workspace") or self.workspace)
        instruction = str(parameters.get("instruction") or "pick up the red block and place it in the bowl")
        target_color = parameters.get("target_color")
        plan = self.planner.plan(instruction, target_color=str(target_color) if target_color else None)
        paths = initialize_workspace(workspace)
        save_plan_document(paths.plan, plan)
        return ToolResponse.success(
            "plan created",
            data={"plan": plan_to_dict(plan), "path": str(paths.plan)},
        )

