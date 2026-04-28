from __future__ import annotations

from typing import Any

import numpy as np


class VLAPolicy:
    """Adapter placeholder for future Vision-Language-Action models.

    A real implementation can wrap OpenVLA, a fine-tuned policy, or a remote
    inference service while keeping the same ``act(observation) -> action`` API.
    """

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError(
            "VLAPolicy is an adapter placeholder. Start with ScriptedPickPlacePolicy "
            "or implement this class with a VLA inference backend."
        )

