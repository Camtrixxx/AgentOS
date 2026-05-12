from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agent.scripted_policy import ScriptedPickPlacePolicy


class RLPolicy:
    """Policy wrapper for RL baselines/checkpoints.

    `backend='scripted'` is a deterministic sanity baseline. `backend='random'`
    is useful for smoke tests. `backend='sb3'` expects stable-baselines3 to be
    installed and a checkpoint path.
    """

    def __init__(
        self,
        *,
        backend: str = "scripted",
        checkpoint: str | Path | None = None,
        seed: int = 0,
    ):
        self.backend = backend
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.rng = np.random.default_rng(seed)
        self._scripted = ScriptedPickPlacePolicy()
        self._model: Any | None = None
        if backend == "sb3":
            self._load_sb3()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self.backend == "scripted":
            return self._scripted.act(observation)
        if self.backend == "random":
            xy = self.rng.uniform(-0.06, 0.06, size=2)
            gripper = self.rng.choice([-1.0, 1.0], size=1)
            return np.concatenate([xy, gripper]).astype(float)
        if self.backend == "sb3":
            if self._model is None:
                raise RuntimeError("SB3 model is not loaded")
            from learning.features import extract_state_features

            action, _ = self._model.predict(extract_state_features(observation), deterministic=True)
            return np.asarray(action, dtype=float)
        raise ValueError(f"Unsupported RL backend {self.backend!r}")

    def _load_sb3(self) -> None:
        if self.checkpoint is None:
            raise ValueError("checkpoint is required for backend='sb3'")
        try:
            from stable_baselines3 import PPO
        except Exception as exc:
            raise RuntimeError("stable-baselines3 is required for backend='sb3'") from exc
        self._model = PPO.load(str(self.checkpoint))

