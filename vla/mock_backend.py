from __future__ import annotations

import numpy as np

from hal.vla_adapter import VLAAction, VLAObservation


class MockVLABackend:
    """Deterministic fake VLA backend for validating integration.

    It mimics the interface of a real VLA model while using state fields to
    produce a reliable pick-and-place action. This lets the rest of the system
    be tested before OpenVLA, LeRobot, or a remote model server is connected.
    """

    name = "mock_vla"

    def __init__(self, step_size: float = 0.06, reach_tolerance: float = 0.04):
        self.step_size = step_size
        self.reach_tolerance = reach_tolerance

    def predict(self, observation: VLAObservation) -> VLAAction:
        state = observation.state
        target_color = self._parse_target_color(observation.instruction)
        target_name = self._find_object_by_color(state["objects"], target_color)
        target_pos = np.asarray(state["objects"][target_name]["position"], dtype=float)
        bowl_pos = np.asarray(state["receptacles"]["bowl"], dtype=float)
        ee_pos = np.asarray(state["ee_position"], dtype=float)
        held_object = state["held_object"]
        gripper_closed = bool(state["gripper_closed"])

        if held_object is None and not gripper_closed:
            if self._distance(ee_pos, target_pos) > self.reach_tolerance:
                return self._move_toward(ee_pos, target_pos, gripper=-1.0)
            return VLAAction(ee_delta=np.zeros(2, dtype=float), gripper=1.0, raw={"phase": "grasp"})

        if held_object is not None:
            if self._distance(ee_pos, bowl_pos) > self.reach_tolerance:
                return self._move_toward(ee_pos, bowl_pos, gripper=1.0)
            return VLAAction(ee_delta=np.zeros(2, dtype=float), gripper=-1.0, raw={"phase": "release"})

        return VLAAction(ee_delta=np.zeros(2, dtype=float), gripper=-1.0, raw={"phase": "idle"})

    def _move_toward(self, current: np.ndarray, target: np.ndarray, gripper: float) -> VLAAction:
        delta = target - current
        norm = np.linalg.norm(delta)
        if norm > self.step_size:
            delta = delta / norm * self.step_size
        return VLAAction(ee_delta=delta.astype(float), gripper=gripper, raw={"phase": "move"})

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _find_object_by_color(objects: dict, color: str) -> str:
        for name, obj in objects.items():
            if obj["color"] == color:
                return name
        raise ValueError(f"No object with color {color!r}")

    @staticmethod
    def _parse_target_color(instruction: str) -> str:
        instruction = instruction.lower()
        for color in ("red", "blue", "green"):
            if color in instruction:
                return color
        return "red"

