from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.robosuite_env import RobosuiteEnvAdapter, RobosuiteEnvConfig
from hal.base_driver import BaseDriver
from runtime.environment_io import environment_doc_from_observation


class RobosuiteDriver(BaseDriver):
    """HAL driver backed by robosuite / MuJoCo.

    This driver is optional. Importing the module is cheap, but constructing the
    driver requires robosuite and mujoco to be installed in the environment.
    """

    def __init__(
        self,
        *,
        task_name: str = "Lift",
        robot: str = "Panda",
        seed: int = 0,
        has_offscreen_renderer: bool = False,
        use_camera_obs: bool = False,
        horizon: int = 200,
    ):
        super().__init__()
        self.config = RobosuiteEnvConfig(
            task_name=task_name,
            robot=robot,
            horizon=horizon,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
        )
        self.env = RobosuiteEnvAdapter(self.config, seed=seed)
        self.instruction = "lift the cube"
        self.target_color = "red"
        self.receptacle_name = "bin"
        self.last_observation = self.env.reset(
            instruction=self.instruction,
            target_color=self.target_color,
            receptacle_name=self.receptacle_name,
        )
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict[str, Any] = {"success": False}
        self.connect()

    def get_profile_path(self) -> Path | None:
        return None

    def load_environment(self, environment: dict[str, Any]) -> None:
        if not isinstance(environment, dict):
            return
        task = environment.get("task")
        if not isinstance(task, dict):
            return
        self.instruction = str(task.get("instruction") or self.instruction)
        self.target_color = str(task.get("target_color") or self.target_color)
        self.receptacle_name = str(task.get("receptacle_name") or self.receptacle_name)

    def _execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if action_type == "reset":
            return self._execute_reset(parameters)
        if action_type == "env_step":
            return self._execute_env_step(parameters)
        return {"success": False, "message": f"unknown action_type {action_type!r}", "action_type": action_type}

    def get_environment(self) -> dict[str, Any]:
        return environment_doc_from_observation(
            self.last_observation,
            target_color=self.target_color,
            receptacle_name=self.receptacle_name,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
        )

    def get_capabilities(self) -> dict[str, Any]:
        native_dim = self.env.action_dim
        action_dims = [3, 4]
        if native_dim and native_dim not in action_dims:
            action_dims.append(native_dim)
        return {
            "supported_actions": ["reset", "env_step"],
            "supported_colors": ["red", "blue", "green"],
            "workspace_low": [-2.0, -2.0, -0.2],
            "workspace_high": [2.0, 2.0, 2.0],
            "max_step_delta": 1.0,
            "action_dims": action_dims,
            "native_action_dim": native_dim,
            "receptacles": ["bin", "bowl"],
            "simulator": "robosuite",
            "task_name": self.config.task_name,
            "robot": self.config.robot,
        }

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected(),
            "healthy": self.health_check(),
            "driver_state": self.state.value,
            "capabilities": self.get_capabilities(),
            "step_count": int(self.last_observation.get("step_count", 0)),
            "target_color": self.target_color,
            "last_reward": self.last_reward,
            "last_done": self.last_done,
            "last_success": bool(self.last_info.get("success", False)),
        }

    def _execute_reset(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self.instruction = str(parameters.get("instruction") or self.instruction)
        self.target_color = str(parameters.get("target_color") or self.target_color)
        self.receptacle_name = str(parameters.get("receptacle_name") or self.receptacle_name)
        self.last_observation = self.env.reset(
            instruction=self.instruction,
            target_color=self.target_color,
            receptacle_name=self.receptacle_name,
        )
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {"success": False}
        return {
            "success": True,
            "message": "robosuite environment reset",
            "action_type": "reset",
            "target_color": self.target_color,
            "step_count": 0,
        }

    def _execute_env_step(self, parameters: dict[str, Any]) -> dict[str, Any]:
        raw_action = parameters.get("action")
        if raw_action is None:
            return {"success": False, "message": "missing parameters.action", "action_type": "env_step"}
        try:
            self.last_observation, reward, done, info = self.env.step(raw_action)
        except Exception as exc:
            return {"success": False, "message": str(exc), "action_type": "env_step"}
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = dict(info)
        return {
            "success": True,
            "message": "robosuite environment stepped",
            "action_type": "env_step",
            "reward": self.last_reward,
            "done": self.last_done,
            "info": self.last_info,
            "step_count": int(self.last_observation.get("step_count", 0)),
        }
