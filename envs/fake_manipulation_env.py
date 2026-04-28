from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjectState:
    name: str
    color: str
    position: np.ndarray


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    target_color: str
    receptacle_name: str = "bowl"


@dataclass(frozen=True)
class FakeManipulationConfig:
    workspace_low: np.ndarray
    workspace_high: np.ndarray
    max_steps: int = 80
    step_size: float = 0.06
    grasp_radius: float = 0.08
    place_radius: float = 0.1


class FakeManipulationEnv:
    """Small 2D pick-and-place environment for agent-loop development.

    The environment is intentionally simple. It gives the project a stable API
    for language-conditioned observation/action loops before introducing a real
    simulator such as LIBERO, robosuite, ManiSkill, or Isaac Lab.
    """

    ACTION_DIM = 3

    def __init__(self, config: FakeManipulationConfig | None = None, seed: int = 0):
        self.config = config or FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
        )
        self.rng = np.random.default_rng(seed)
        self.task = TaskSpec("pick up the red block and place it in the bowl", "red")
        self.ee_position = np.zeros(2, dtype=float)
        self.gripper_closed = False
        self.held_object: str | None = None
        self.objects: dict[str, ObjectState] = {}
        self.receptacles: dict[str, np.ndarray] = {}
        self.step_count = 0

    def reset(self, task: TaskSpec | None = None) -> dict[str, Any]:
        self.task = task or self.task
        self.ee_position = np.array([0.0, -0.75], dtype=float)
        self.gripper_closed = False
        self.held_object = None
        self.step_count = 0

        self.objects = {
            "red_block": ObjectState("red_block", "red", np.array([-0.55, 0.15], dtype=float)),
            "blue_block": ObjectState("blue_block", "blue", np.array([0.45, 0.1], dtype=float)),
            "green_block": ObjectState("green_block", "green", np.array([-0.1, 0.45], dtype=float)),
        }
        self.receptacles = {"bowl": np.array([0.55, 0.65], dtype=float)}
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(f"Expected action shape ({self.ACTION_DIM},), got {action.shape}")

        delta = np.clip(action[:2], -self.config.step_size, self.config.step_size)
        self.ee_position = np.clip(
            self.ee_position + delta,
            self.config.workspace_low,
            self.config.workspace_high,
        )
        close_gripper = action[2] > 0.0
        self._update_gripper(close_gripper)
        self.step_count += 1

        success = self._is_success()
        timeout = self.step_count >= self.config.max_steps
        reward = 1.0 if success else -0.01
        done = success or timeout
        info = {"success": success, "timeout": timeout, "held_object": self.held_object}
        return self._observation(), reward, done, info

    def action_space_sample(self) -> np.ndarray:
        xy = self.rng.uniform(-self.config.step_size, self.config.step_size, size=2)
        grip = self.rng.choice([-1.0, 1.0], size=1)
        return np.concatenate([xy, grip])

    def _update_gripper(self, close_gripper: bool) -> None:
        if close_gripper and not self.gripper_closed:
            nearest_name = self._nearest_object_name()
            if nearest_name is not None:
                obj = self.objects[nearest_name]
                if np.linalg.norm(obj.position - self.ee_position) <= self.config.grasp_radius:
                    self.held_object = nearest_name
        elif not close_gripper and self.gripper_closed:
            self.held_object = None

        self.gripper_closed = close_gripper
        if self.held_object is not None:
            obj = self.objects[self.held_object]
            self.objects[self.held_object] = ObjectState(obj.name, obj.color, self.ee_position.copy())

    def _nearest_object_name(self) -> str | None:
        if not self.objects:
            return None
        return min(
            self.objects,
            key=lambda name: float(np.linalg.norm(self.objects[name].position - self.ee_position)),
        )

    def _target_object_name(self) -> str:
        for name, obj in self.objects.items():
            if obj.color == self.task.target_color:
                return name
        raise ValueError(f"No object with color {self.task.target_color!r}")

    def _is_success(self) -> bool:
        target = self.objects[self._target_object_name()]
        receptacle = self.receptacles[self.task.receptacle_name]
        return (
            self.held_object is None
            and not self.gripper_closed
            and np.linalg.norm(target.position - receptacle) <= self.config.place_radius
        )

    def _observation(self) -> dict[str, Any]:
        return {
            "instruction": self.task.instruction,
            "ee_position": self.ee_position.copy(),
            "gripper_closed": self.gripper_closed,
            "held_object": self.held_object,
            "objects": {
                name: {"color": obj.color, "position": obj.position.copy()}
                for name, obj in self.objects.items()
            },
            "receptacles": {name: pos.copy() for name, pos in self.receptacles.items()},
            "step_count": self.step_count,
        }

