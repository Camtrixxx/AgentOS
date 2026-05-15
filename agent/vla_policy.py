from __future__ import annotations

from typing import Any

import numpy as np

from hal.vla_adapter import FakeEnvVLAAdapter
from vla.base import VLABackend
from vla.mock_backend import MockVLABackend


class VLAPolicy:
    """Policy wrapper for Vision-Language-Action backends.

    The backend can be a mock implementation, a local VLA model, or a remote
    inference service. The policy keeps the Policy protocol stable.
    """

    def __init__(
        self,
        backend: VLABackend | None = None,
        adapter: FakeEnvVLAAdapter | None = None,
    ):
        self.backend = backend or MockVLABackend()
        self.adapter = adapter or FakeEnvVLAAdapter()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        vla_observation = self.adapter.observation_to_vla(observation)
        vla_action = self.backend.predict(vla_observation)
        return self.adapter.action_from_vla(vla_action)
