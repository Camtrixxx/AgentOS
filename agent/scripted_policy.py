from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.task_utils import find_object_by_color, parse_target_color


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    step_size: float = 0.06
    reach_tolerance: float = 0.04


class ScriptedPickPlacePolicy:
    """Rule-based language-conditioned policy for the fake environment."""

    def __init__(self, config: ScriptedPolicyConfig | None = None):
        self.config = config or ScriptedPolicyConfig()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        target_color = parse_target_color(observation["instruction"])
        target_name = find_object_by_color(observation["objects"], target_color)
        target_pos = observation["objects"][target_name]["position"]
        bowl_pos = observation["receptacles"]["bowl"]
        ee_pos = observation["ee_position"]
        held_object = observation["held_object"]

        if held_object is None and not observation["gripper_closed"]:
            if self._distance(ee_pos, target_pos) > self.config.reach_tolerance:
                return self._move_toward(ee_pos, target_pos, grip=-1.0)
            return np.array([0.0, 0.0, 1.0], dtype=float)

        if held_object is not None:
            if self._distance(ee_pos, bowl_pos) > self.config.reach_tolerance:
                return self._move_toward(ee_pos, bowl_pos, grip=1.0)
            return np.array([0.0, 0.0, -1.0], dtype=float)

        return np.array([0.0, 0.0, -1.0], dtype=float)

    def _move_toward(self, current: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
        delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        norm = np.linalg.norm(delta)
        if norm > self.config.step_size:
            delta = delta / norm * self.config.step_size
        return np.array([delta[0], delta[1], grip], dtype=float)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


