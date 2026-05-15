from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.robosuite_scripted_policy import RobosuiteLiftPolicy
from runtime.repository import WorkspaceRepository, resolve_repo
from tools.embodied_tools import StepEnvTool
from tools.response import ToolResponse


class RobosuiteLiftLoopTool:
    name = "robosuite_lift_loop"
    description = "Run a scripted robosuite Lift policy through ACTION.md and the watchdog."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: Any | None = None,
        policy: RobosuiteLiftPolicy | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.step_tool = StepEnvTool(workspace, driver=driver)
        self.policy = policy or RobosuiteLiftPolicy()

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        max_steps = int(parameters.get("max_steps", 120))
        records: list[dict[str, Any]] = []

        for _ in range(max_steps):
            environment = repo.get_environment()
            episode = environment.get("episode", {}) if isinstance(environment.get("episode"), dict) else {}
            if bool(episode.get("success", False)) or bool(episode.get("done", False)):
                return ToolResponse.success(
                    "robosuite lift loop already complete",
                    data={"success": bool(episode.get("success", False)), "step_records": records},
                )

            action = self.policy.act(environment)
            response = self.step_tool.run({"workspace": repo.paths.root, "action": action})
            updated_environment = repo.get_environment()
            updated_episode = updated_environment.get("episode", {}) if isinstance(updated_environment, dict) else {}
            record = {
                "stage": self.policy.last_stage,
                "action": action,
                "tool_status": response.status.value,
                "tool_text": response.text,
                "step_count": updated_episode.get("step_count"),
                "reward": updated_episode.get("last_reward"),
                "success": bool(updated_episode.get("success", False)),
                "done": bool(updated_episode.get("done", False)),
            }
            if response.error is not None:
                record["error"] = response.error
            records.append(record)
            if response.error is not None:
                return ToolResponse.failure(
                    response.error.get("code", "robosuite_lift_failed"),
                    response.error.get("message", response.text),
                    data={"success": False, "step_records": records, "environment": updated_environment},
                )
            if bool(updated_episode.get("success", False)) or bool(updated_episode.get("done", False)):
                return ToolResponse.success(
                    "robosuite lift loop completed",
                    data={
                        "success": bool(updated_episode.get("success", False)),
                        "step_records": records,
                        "environment": updated_environment,
                    },
                )

        environment = repo.get_environment()
        return ToolResponse.failure(
            "robosuite_lift_timeout",
            f"robosuite lift loop reached max_steps={max_steps}",
            data={"success": False, "step_records": records, "environment": environment},
        )
