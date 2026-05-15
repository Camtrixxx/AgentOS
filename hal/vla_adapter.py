from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VLAObservation:
    """Standard input payload for VLA-style backends."""

    image: np.ndarray
    instruction: str
    state: dict[str, Any]


@dataclass(frozen=True)
class VLAAction:
    """Standard output payload from VLA-style backends."""

    ee_delta: np.ndarray
    gripper: float
    raw: Any | None = None


class FakeEnvVLAAdapter:
    """Translate FakeManipulationEnv observations/actions to a VLA contract."""

    def observation_to_vla(self, observation: dict[str, Any]) -> VLAObservation:
        image = observation.get("image")
        if image is None:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        return VLAObservation(
            image=np.asarray(image, dtype=np.uint8),
            instruction=str(observation["instruction"]),
            state={
                "ee_position": np.asarray(observation["ee_position"], dtype=float),
                "gripper_closed": bool(observation["gripper_closed"]),
                "held_object": observation["held_object"],
                "objects": observation["objects"],
                "receptacles": observation["receptacles"],
                "step_count": int(observation["step_count"]),
            },
        )

    def action_from_vla(self, action: VLAAction) -> np.ndarray:
        ee_delta = np.asarray(action.ee_delta, dtype=float)
        if ee_delta.shape != (2,):
            raise ValueError(f"Expected ee_delta shape (2,), got {ee_delta.shape}")
        return np.array([ee_delta[0], ee_delta[1], float(action.gripper)], dtype=float)

