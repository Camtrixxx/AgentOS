from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


@dataclass
class RobosuiteLiftPolicy:
    """Small finite-state scripted policy for robosuite Lift.

    The policy emits compact AgentOS actions: [dx, dy, dz, gripper].
    RobosuiteEnvAdapter expands these into the native controller action.
    """

    max_delta: float = 1.0
    position_gain: float = 5.0
    hover_offset: float = 0.18
    grasp_offset: float = -0.03
    lift_offset: float = 0.28
    xy_tolerance: float = 0.025
    z_tolerance: float = 0.025
    close_steps: int = 25
    last_stage: str = "init"
    _close_count: int = 0

    def act(self, environment: dict[str, Any]) -> list[float]:
        ee = _ee_position(environment)
        cube = _object_position(environment, "cube")
        if cube is None:
            self.last_stage = "missing_cube"
            return [0.0, 0.0, 0.0, GRIPPER_OPEN]

        hover = cube.copy()
        hover[2] += self.hover_offset
        grasp = cube.copy()
        grasp[2] += self.grasp_offset
        lift = cube.copy()
        lift[2] += self.lift_offset

        xy_error = float(np.linalg.norm(cube[:2] - ee[:2]))
        if self._close_count >= self.close_steps:
            if ee[2] < lift[2] - self.z_tolerance:
                self.last_stage = "lift"
                return self._action_toward(ee, lift, GRIPPER_CLOSE)
            self.last_stage = "hold"
            return [0.0, 0.0, 0.0, GRIPPER_CLOSE]

        if xy_error > self.xy_tolerance:
            self._close_count = 0
            self.last_stage = "move_above_cube"
            return self._action_toward(ee, hover, GRIPPER_OPEN)

        if ee[2] > grasp[2] + self.z_tolerance:
            self._close_count = 0
            self.last_stage = "descend"
            return self._action_toward(ee, grasp, GRIPPER_OPEN)

        if self._close_count < self.close_steps:
            self._close_count += 1
            self.last_stage = "close_gripper"
            return [0.0, 0.0, 0.0, GRIPPER_CLOSE]

    def _action_toward(self, ee: np.ndarray, target: np.ndarray, gripper: float) -> list[float]:
        command = np.clip((target[:3] - ee[:3]) * self.position_gain, -self.max_delta, self.max_delta)
        return [float(command[0]), float(command[1]), float(command[2]), float(gripper)]


def _ee_position(environment: dict[str, Any]) -> np.ndarray:
    robot = environment.get("robot", {}) if isinstance(environment.get("robot"), dict) else {}
    return _as_vec3(robot.get("ee_position"), default=[0.0, 0.0, 0.0])


def _object_position(environment: dict[str, Any], name: str) -> np.ndarray | None:
    objects = environment.get("objects", {}) if isinstance(environment.get("objects"), dict) else {}
    payload = objects.get(name)
    if not isinstance(payload, dict):
        return None
    return _as_vec3(payload.get("position"), default=None)


def _as_vec3(value: Any, *, default: list[float] | None) -> np.ndarray | None:
    if value is None:
        return None if default is None else np.asarray(default, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3:
        return None if default is None else np.asarray(default, dtype=float)
    return array[:3]
