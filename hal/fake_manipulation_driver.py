from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, ObjectState, TaskSpec
from hal.base_driver import BaseDriver
from runtime.environment_io import environment_doc_from_observation


class FakeManipulationDriver(BaseDriver):
    """HAL driver backed by FakeManipulationEnv."""

    def __init__(
        self,
        *,
        seed: int = 0,
        include_image: bool = False,
        randomize_layout: bool = False,
        max_steps: int = 80,
    ):
        self.config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            include_image=include_image,
            randomize_layout=randomize_layout,
            max_steps=max_steps,
        )
        self.env = FakeManipulationEnv(config=self.config, seed=seed)
        self.last_observation = self.env.reset()
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict[str, Any] = {"success": False}

    def get_profile_path(self) -> Path | None:
        return None

    def load_environment(self, environment: dict[str, Any]) -> None:
        if not isinstance(environment, dict):
            return
        task = environment.get("task")
        if not isinstance(task, dict):
            task = {}
        instruction = str(task.get("instruction") or "pick up the red block and place it in the bowl")
        target_color = str(task.get("target_color") or "red")
        receptacle_name = str(task.get("receptacle_name") or "bowl")
        self.env.task = TaskSpec(instruction=instruction, target_color=target_color, receptacle_name=receptacle_name)

        robot = environment.get("robot")
        if isinstance(robot, dict):
            self.env.ee_position = np.asarray(robot.get("ee_position", [0.0, -0.75]), dtype=float)
            self.env.gripper_closed = bool(robot.get("gripper_closed", False))
            self.env.held_object = robot.get("held_object")

        objects = environment.get("objects")
        if isinstance(objects, dict) and objects:
            self.env.objects = {
                name: ObjectState(
                    name=name,
                    color=str(payload.get("color") or "unknown"),
                    position=np.asarray(payload.get("position", [0.0, 0.0]), dtype=float),
                )
                for name, payload in objects.items()
                if isinstance(payload, dict)
            }

        receptacles = environment.get("receptacles")
        if isinstance(receptacles, dict) and receptacles:
            self.env.receptacles = {
                name: np.asarray(
                    payload.get("position", payload) if isinstance(payload, dict) else payload,
                    dtype=float,
                )
                for name, payload in receptacles.items()
            }

        episode = environment.get("episode")
        if isinstance(episode, dict):
            self.env.step_count = int(episode.get("step_count", 0))
            self.last_reward = float(episode.get("last_reward", 0.0))
            self.last_done = bool(episode.get("done", False))
            last_info = episode.get("last_info", {})
            self.last_info = dict(last_info) if isinstance(last_info, dict) else {"success": False}

        self.last_observation = self.env._observation()
        if not self.env.objects or not self.env.receptacles:
            self.reset(instruction=instruction, target_color=target_color, receptacle_name=receptacle_name)

    def execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if action_type == "reset":
            return self._execute_reset(parameters)
        if action_type == "env_step":
            return self._execute_env_step(parameters)
        return {
            "success": False,
            "message": f"unknown action_type {action_type!r}",
            "action_type": action_type,
        }

    def get_environment(self) -> dict[str, Any]:
        return environment_doc_from_observation(
            self.last_observation,
            target_color=self.env.task.target_color,
            receptacle_name=self.env.task.receptacle_name,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
        )

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "connected": True,
            "healthy": True,
            "step_count": int(self.last_observation.get("step_count", 0)),
            "target_color": self.env.task.target_color,
            "last_reward": self.last_reward,
            "last_done": self.last_done,
            "last_success": bool(self.last_info.get("success", False)),
        }

    def reset(self, *, instruction: str, target_color: str, receptacle_name: str = "bowl") -> dict[str, Any]:
        task = TaskSpec(instruction=instruction, target_color=target_color, receptacle_name=receptacle_name)
        self.last_observation = self.env.reset(task)
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {"success": False}
        return self.last_observation

    def _execute_reset(self, parameters: dict[str, Any]) -> dict[str, Any]:
        instruction = str(parameters.get("instruction") or "pick up the red block and place it in the bowl")
        target_color = str(parameters.get("target_color") or "red")
        receptacle_name = str(parameters.get("receptacle_name") or "bowl")
        self.reset(instruction=instruction, target_color=target_color, receptacle_name=receptacle_name)
        return {
            "success": True,
            "message": "environment reset",
            "action_type": "reset",
            "target_color": target_color,
            "step_count": 0,
        }

    def _execute_env_step(self, parameters: dict[str, Any]) -> dict[str, Any]:
        raw_action = parameters.get("action")
        if raw_action is None:
            return {"success": False, "message": "missing parameters.action", "action_type": "env_step"}

        action = np.asarray(raw_action, dtype=float)
        if action.shape != (3,):
            return {
                "success": False,
                "message": f"expected action shape (3,), got {tuple(action.shape)}",
                "action_type": "env_step",
            }

        self.last_observation, reward, done, info = self.env.step(action)
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = dict(info)
        return {
            "success": True,
            "message": "environment stepped",
            "action_type": "env_step",
            "reward": self.last_reward,
            "done": self.last_done,
            "info": self.last_info,
            "step_count": int(self.last_observation["step_count"]),
        }
