from __future__ import annotations

from typing import Protocol

from hal.vla_adapter import VLAAction, VLAObservation


class VLABackend(Protocol):
    """Minimal backend contract for Vision-Language-Action inference."""

    name: str

    def predict(self, observation: VLAObservation) -> VLAAction:
        ...

