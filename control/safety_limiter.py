from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyConfig:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_delta_per_step: float = 0.02
    command_timeout_s: float = 0.5


class SafetyLimiter:
    """Clamp joint commands and limit per-frame motion."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self._last_command: np.ndarray | None = None

    def reset(self, current_position: np.ndarray | None = None) -> None:
        self._last_command = None if current_position is None else np.asarray(current_position, dtype=float)

    def limit(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if target.shape != self.config.joint_lower.shape:
            raise ValueError(f"Expected command shape {self.config.joint_lower.shape}, got {target.shape}")

        clipped = np.clip(target, self.config.joint_lower, self.config.joint_upper)
        if self._last_command is None:
            self._last_command = clipped
            return clipped.copy()

        delta = np.clip(
            clipped - self._last_command,
            -self.config.max_delta_per_step,
            self.config.max_delta_per_step,
        )
        limited = self._last_command + delta
        self._last_command = limited
        return limited.copy()

