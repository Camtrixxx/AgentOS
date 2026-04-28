from __future__ import annotations

from typing import Any

import numpy as np


COLORS = ("red", "blue", "green")
FEATURE_DIM = 15


def parse_target_color(instruction: str) -> str:
    instruction = instruction.lower()
    for color in COLORS:
        if color in instruction:
            return color
    return "red"


def extract_state_features(observation: dict[str, Any]) -> np.ndarray:
    """Convert a fake-env observation into a compact numeric state vector.

    Feature layout:
    - ee xy: 2
    - target object xy: 2
    - bowl xy: 2
    - target relative to ee: 2
    - bowl relative to ee: 2
    - gripper closed flag: 1
    - holding target flag: 1
    - target color one-hot: 3
    """

    instruction = observation["instruction"]
    target_color = parse_target_color(instruction)
    target_name = find_object_by_color(observation, target_color)

    ee_pos = np.asarray(observation["ee_position"], dtype=np.float32)
    target_pos = np.asarray(observation["objects"][target_name]["position"], dtype=np.float32)
    bowl_pos = np.asarray(observation["receptacles"]["bowl"], dtype=np.float32)
    gripper_closed = np.asarray([float(observation["gripper_closed"])], dtype=np.float32)
    holding_target = np.asarray([float(observation["held_object"] == target_name)], dtype=np.float32)
    color_one_hot = np.asarray([float(color == target_color) for color in COLORS], dtype=np.float32)

    features = np.concatenate(
        [
            ee_pos,
            target_pos,
            bowl_pos,
            target_pos - ee_pos,
            bowl_pos - ee_pos,
            gripper_closed,
            holding_target,
            color_one_hot,
        ]
    ).astype(np.float32)
    if features.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected feature shape ({FEATURE_DIM},), got {features.shape}")
    return features


def find_object_by_color(observation: dict[str, Any], color: str) -> str:
    for name, obj in observation["objects"].items():
        if obj["color"] == color:
            return name
    raise ValueError(f"No object with color {color!r}")
