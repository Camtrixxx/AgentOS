from __future__ import annotations

from typing import Any

import numpy as np

from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec
from learning.features import FEATURE_DIM, extract_state_features


class FakeManipulationGymEnv:
    """Small Gymnasium-style adapter without requiring gymnasium as a dependency."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        randomize_layout: bool = False,
        include_image: bool = False,
        seed: int = 0,
        max_steps: int = 80,
    ):
        self.config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            randomize_layout=randomize_layout,
            include_image=include_image,
            max_steps=max_steps,
        )
        self.env = FakeManipulationEnv(config=self.config, seed=seed)
        self.task = TaskSpec("pick up the red block and place it in the bowl", "red")
        self.last_observation: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.env = FakeManipulationEnv(config=self.config, seed=seed)
        options = options or {}
        target_color = str(options.get("target_color") or self.task.target_color)
        instruction = str(options.get("instruction") or f"pick up the {target_color} block and place it in the bowl")
        self.task = TaskSpec(instruction, target_color)
        self.last_observation = self.env.reset(self.task)
        return extract_state_features(self.last_observation).astype(np.float32), {"observation": self.last_observation}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        self.last_observation = observation
        terminated = bool(info.get("success", False))
        truncated = bool(info.get("timeout", False)) and not terminated
        return extract_state_features(observation).astype(np.float32), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.env.render_rgb()

    @property
    def observation_dim(self) -> int:
        return FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return self.env.ACTION_DIM

