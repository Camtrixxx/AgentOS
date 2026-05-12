from __future__ import annotations

from pathlib import Path
from typing import Any

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.action_validator import validate_action
from runtime.action_queue import append_action, load_action_document, save_action_document
from runtime.environment_io import load_environment_document
from runtime.watchdog import poll_once
from runtime.workspace import initialize_workspace
from tools.response import ToolResponse


class ReadEnvironmentTool:
    name = "read_environment"
    description = "Read the current embodied workspace ENVIRONMENT.md document."

    def __init__(self, workspace: str | Path = "workspace"):
        self.workspace = Path(workspace)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        workspace = Path(parameters.get("workspace") or self.workspace)
        paths = initialize_workspace(workspace)
        environment = load_environment_document(paths.environment)
        return ToolResponse.success(
            "environment loaded",
            data={"environment": environment, "path": str(paths.environment)},
        )


class AppendActionTool:
    name = "append_action"
    description = "Append a pending embodied action to ACTION.md."

    def __init__(self, workspace: str | Path = "workspace"):
        self.workspace = Path(workspace)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        workspace = Path(parameters.get("workspace") or self.workspace)
        action_type = str(parameters.get("action_type") or "").strip()
        if not action_type:
            return ToolResponse.failure("invalid_action", "parameters.action_type is required")
        action_parameters = parameters.get("parameters", {})
        if not isinstance(action_parameters, dict):
            return ToolResponse.failure("invalid_parameters", "parameters.parameters must be an object")

        paths = initialize_workspace(workspace)
        environment = load_environment_document(paths.environment)
        validation = validate_action(action_type, action_parameters, environment)
        if not validation.valid:
            return ToolResponse.failure(
                "critic_rejected_action",
                validation.reason,
                data={"action_type": action_type, "parameters": action_parameters},
            )
        document = load_action_document(paths.action)
        updated = append_action(document, action_type=action_type, parameters=action_parameters)
        save_action_document(paths.action, updated)
        action = updated["actions"][-1]
        return ToolResponse.success(f"queued action {action['id']}", data={"action": action, "path": str(paths.action)})


class RunWatchdogOnceTool:
    name = "run_watchdog_once"
    description = "Execute the first pending ACTION.md item through the fake manipulation HAL driver."

    def __init__(self, workspace: str | Path = "workspace", driver: FakeManipulationDriver | None = None):
        self.workspace = Path(workspace)
        self.driver = driver

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        workspace = Path(parameters.get("workspace") or self.workspace)
        paths = initialize_workspace(workspace)
        driver = self.driver or FakeManipulationDriver(
            seed=int(parameters.get("seed", 0)),
            include_image=bool(parameters.get("include_image", False)),
            randomize_layout=bool(parameters.get("randomize_layout", False)),
            max_steps=int(parameters.get("max_steps", 80)),
        )
        driver.load_environment(load_environment_document(paths.environment))
        result = poll_once(driver, paths)
        environment = load_environment_document(paths.environment)
        if result is None:
            return ToolResponse.partial(
                "no pending action",
                data={"result": None, "environment": environment, "path": str(paths.action)},
            )
        if not result.get("success", False):
            return ToolResponse.failure(
                "action_failed",
                str(result.get("message") or "action failed"),
                data={"result": result, "environment": environment},
            )
        return ToolResponse.success("watchdog executed action", data={"result": result, "environment": environment})


class ResetTaskTool:
    name = "reset_task"
    description = "Queue and execute a reset action for the fake manipulation task."

    def __init__(self, workspace: str | Path = "workspace", driver: FakeManipulationDriver | None = None):
        self.workspace = Path(workspace)
        self.append_tool = AppendActionTool(workspace)
        self.watchdog_tool = RunWatchdogOnceTool(workspace, driver=driver)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        target_color = str(parameters.get("target_color") or _infer_target_color(str(parameters.get("instruction") or "")))
        instruction = str(parameters.get("instruction") or f"pick up the {target_color} block and place it in the bowl")
        receptacle_name = str(parameters.get("receptacle_name") or "bowl")
        workspace = Path(parameters.get("workspace") or self.workspace)
        queued = self.append_tool.run(
            {
                "workspace": workspace,
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
        return self.watchdog_tool.run({"workspace": workspace})


class StepEnvTool:
    name = "step_env"
    description = "Queue and execute one fake manipulation env_step action."

    def __init__(self, workspace: str | Path = "workspace", driver: FakeManipulationDriver | None = None):
        self.workspace = Path(workspace)
        self.append_tool = AppendActionTool(workspace)
        self.watchdog_tool = RunWatchdogOnceTool(workspace, driver=driver)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        action = parameters.get("action")
        if action is None:
            return ToolResponse.failure("invalid_action", "parameters.action is required")
        workspace = Path(parameters.get("workspace") or self.workspace)
        queued = self.append_tool.run(
            {
                "workspace": workspace,
                "action_type": "env_step",
                "parameters": {"action": action},
            }
        )
        if queued.error is not None:
            return queued
        return self.watchdog_tool.run({"workspace": workspace})


def _infer_target_color(instruction: str) -> str:
    lowered = instruction.lower()
    for color in ("red", "blue", "green"):
        if color in lowered:
            return color
    return "red"
