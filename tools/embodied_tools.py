from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.task_utils import parse_target_color
from agent.scripted_policy import ScriptedPickPlacePolicy
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.action_queue import append_action
from runtime.action_validator import validate_action
from runtime.environment_io import observation_from_environment_doc
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
        def append(document: dict[str, Any]) -> dict[str, Any]:
            updated = append_action(document, action_type=action_type, parameters=action_parameters)
            document.clear()
            document.update(updated)
            return updated["actions"][-1]

        action = repo.update_actions(append)
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


class ScriptedPickPlaceLoopTool:
    name = "scripted_pick_place_loop"
    description = "Run a scripted pick-place policy through ACTION.md and the watchdog until success or timeout."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: FakeManipulationDriver | None = None,
        policy: Any | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.step_tool = StepEnvTool(workspace, driver=driver)
        self.policy = policy or ScriptedPickPlacePolicy()

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        max_steps = int(parameters.get("max_steps", 80))
        records: list[dict[str, Any]] = []

        for _ in range(max_steps):
            environment = repo.get_environment()
            episode = environment.get("episode", {}) if isinstance(environment, dict) else {}
            if bool(episode.get("success", False)) or bool(episode.get("done", False)):
                return ToolResponse.success(
                    "scripted loop already complete",
                    data={"success": bool(episode.get("success", False)), "step_records": records},
                )

            observation = observation_from_environment_doc(environment)
            action = self.policy.act(observation).tolist()
            response = self.step_tool.run({"workspace": repo.paths.root, "action": action})
            updated_environment = repo.get_environment()
            updated_episode = updated_environment.get("episode", {}) if isinstance(updated_environment, dict) else {}
            record = {
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
                    response.error.get("code", "scripted_loop_failed"),
                    response.error.get("message", response.text),
                    data={"success": False, "step_records": records, "environment": updated_environment},
                )
            if bool(updated_episode.get("success", False)) or bool(updated_episode.get("done", False)):
                return ToolResponse.success(
                    "scripted loop completed",
                    data={
                        "success": bool(updated_episode.get("success", False)),
                        "step_records": records,
                        "environment": updated_environment,
                    },
                )

        environment = repo.get_environment()
        return ToolResponse.failure(
            "scripted_loop_timeout",
            f"scripted loop reached max_steps={max_steps}",
            data={"success": False, "step_records": records, "environment": environment},
        )
