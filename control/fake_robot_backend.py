from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    joint_position: np.ndarray
    frame_id: int = 0


class FakeRobotBackend:
    """A deterministic robot backend for development without hardware."""

    def __init__(self, joint_names: list[str]):
        self.joint_names = joint_names
        self.state = RobotState(joint_position=np.zeros(len(joint_names), dtype=float))
        self.command_log: list[np.ndarray] = []

    def send_joint_command(self, joint_position: np.ndarray) -> RobotState:
        joint_position = np.asarray(joint_position, dtype=float)
        if joint_position.shape != self.state.joint_position.shape:
            raise ValueError(
                f"Expected {self.state.joint_position.shape[0]} joints, got {joint_position.shape[0]}"
            )
        self.state = RobotState(joint_position=joint_position.copy(), frame_id=self.state.frame_id + 1)
        self.command_log.append(joint_position.copy())
        return self.state

    def get_state(self) -> RobotState:
        return RobotState(joint_position=self.state.joint_position.copy(), frame_id=self.state.frame_id)

