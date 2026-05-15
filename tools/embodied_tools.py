from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.task_utils import parse_target_color
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.action_queue import append_action
from runtime.action_validator import validate_action
from runtime.repository import WorkspaceRepository, resolve_repo
from runtime.watchdog import poll_once
from tools.response import ToolResponse


class ReadEnvironmentTool:
    name = "read_environment"
    description = "Read the current embodied workspace ENVIRONMENT.md document."

    def __init__(self, workspace: str | Path | WorkspaceRepository = "workspace"):
        self._repo_params: str | Path | WorkspaceRepository = workspace

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        environment = repo.get_environment()
        return ToolResponse.success(
            "environment loaded",
            data={"environment": environment, "path": str(repo.paths.environment)},
        )


class AppendActionTool:
    name = "append_action"
    description = "Append a pending embodied action to ACTION.md."

    def __init__(self, workspace: str | Path | WorkspaceRepository = "workspace"):
        self._repo_params: str | Path | WorkspaceRepository = workspace

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        action_type = str(parameters.get("action_type") or "").strip()
        if not action_type:
            return ToolResponse.failure("invalid_action", "parameters.action_type is required")
        action_parameters = parameters.get("parameters", {})
        if not isinstance(action_parameters, dict):
            return ToolResponse.failure("invalid_parameters", "parameters.parameters must be an object")

        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        environment = repo.get_environment()
        validation = validate_action(action_type, action_parameters, environment)
        if not validation.valid:
            return ToolResponse.failure(
                "critic_rejected_action",
                validation.reason,
                data={"action_type": action_type, "parameters": action_parameters},
            )
        document = repo.get_actions()
        updated = append_action(document, action_type=action_type, parameters=action_parameters)
        repo.save_actions(updated)
        action = updated["actions"][-1]
        return ToolResponse.success(
            f"queued action {action['id']}",
            data={"action": action, "path": str(repo.paths.action)},
        )


class RunWatchdogOnceTool:
    name = "run_watchdog_once"
    description = "Execute the first pending ACTION.md item through the fake manipulation HAL driver."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: FakeManipulationDriver | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.driver = driver

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        driver = self.driver or FakeManipulationDriver(
            seed=int(parameters.get("seed", 0)),
            include_image=bool(parameters.get("include_image", False)),
            randomize_layout=bool(parameters.get("randomize_layout", False)),
            max_steps=int(parameters.get("max_steps", 80)),
        )
        driver.load_environment(repo.get_environment())
        result = poll_once(driver, repo)
        environment = repo.get_environment()
        if result is None:
            return ToolResponse.partial(
                "no pending action",
                data={"result": None, "environment": environment, "path": str(repo.paths.action)},
            )
        if not result.get("success", False):
            return ToolResponse.failure(
                "action_failed",
                str(result.get("message") or "action failed"),
                data={"result": result, "environment": environment},
            )
        return ToolResponse.success(
            "watchdog executed action",
            data={"result": result, "environment": environment},
        )


class ResetTaskTool:
    name = "reset_task"
    description = "Queue and execute a reset action for the fake manipulation task."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: FakeManipulationDriver | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.append_tool = AppendActionTool(workspace)
        self.watchdog_tool = RunWatchdogOnceTool(workspace, driver=driver)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        target_color = str(parameters.get("target_color") or parse_target_color(str(parameters.get("instruction") or "")))
        instruction = str(parameters.get("instruction") or f"pick up the {target_color} block and place it in the bowl")
        receptacle_name = str(parameters.get("receptacle_name") or "bowl")
        queued = self.append_tool.run(
            {
                "workspace": repo.paths.root,
                "action_type": "reset",
                "parameters": {
                    "instruction": instruction,
                    "target_color": target_color,
                    "receptacle_name": receptacle_name,
                },
            }
        )
        if queued.error is not None:
            return queued
        return self.watchdog_tool.run({"workspace": repo.paths.root})


class StepEnvTool:
    name = "step_env"
    description = "Queue and execute one fake manipulation env_step action."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: FakeManipulationDriver | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.append_tool = AppendActionTool(workspace)
        self.watchdog_tool = RunWatchdogOnceTool(workspace, driver=driver)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        action = parameters.get("action")
        if action is None:
            return ToolResponse.failure("invalid_action", "parameters.action is required")
        queued = self.append_tool.run(
            {
                "workspace": repo.paths.root,
                "action_type": "env_step",
                "parameters": {"action": action},
            }
        )
        if queued.error is not None:
            return queued
        return self.watchdog_tool.run({"workspace": repo.paths.root})
