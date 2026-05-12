from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from adapters.vla_adapter import VLAAction, VLAObservation
from learning.features import extract_state_features
from vla.mock_backend import MockVLABackend


class SmolVLABackend:
    """SmolVLA backend adapter for the VLABackend protocol.

    If LeRobot/SmolVLA is not installed or `dry_run=True`, this backend falls
    back to a deterministic MockVLABackend while preserving the integration
    point and metadata shape.
    """

    name = "smolvla"

    def __init__(
        self,
        model_path: str | Path = "lerobot/smolvla_base",
        *,
        device: str = "cpu",
        dry_run: bool = False,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.dry_run = dry_run
        self._mock = MockVLABackend()
        self._policy: Any | None = None
        self._load_error: str | None = None
        if not dry_run:
            self._try_load_policy()

    def predict(self, observation: VLAObservation) -> VLAAction:
        if self._policy is None:
            action = self._mock.predict(observation)
            raw = dict(action.raw or {})
            raw.update(
                {
                    "backend": self.name,
                    "mode": "mock_fallback",
                    "model_path": self.model_path,
                    "load_error": self._load_error,
                }
            )
            return VLAAction(ee_delta=action.ee_delta, gripper=action.gripper, raw=raw)

        batch = self._make_lerobot_batch(observation)
        selected = self._policy.select_action(batch)
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().numpy()
        action_array = np.asarray(selected, dtype=float).reshape(-1)
        if action_array.shape[0] < 3:
            raise ValueError(f"SmolVLA action must have at least 3 values, got {action_array.shape}")
        return VLAAction(ee_delta=action_array[:2], gripper=float(action_array[2]), raw={"backend": self.name})

    def _try_load_policy(self) -> None:
        SmolVLAPolicy = self._import_smolvla_policy()
        if SmolVLAPolicy is None:
            return
        try:
            self._policy = SmolVLAPolicy.from_pretrained(self.model_path)
            to_method = getattr(self._policy, "to", None)
            if callable(to_method):
                self._policy = to_method(self.device)
            eval_method = getattr(self._policy, "eval", None)
            if callable(eval_method):
                eval_method()
        except Exception as exc:
            self._load_error = f"failed to load SmolVLA model {self.model_path!r}: {type(exc).__name__}: {exc}"
            self._policy = None

    def _import_smolvla_policy(self) -> Any | None:
        import_errors: list[str] = []
        module_paths = [
            "lerobot.policies.smolvla.modeling_smolvla",
            "lerobot.policies.smolvla",
            "lerobot.common.policies.smolvla.modeling_smolvla",
        ]
        for module_path in module_paths:
            try:
                module = __import__(module_path, fromlist=["SmolVLAPolicy"])
                return getattr(module, "SmolVLAPolicy")
            except Exception as exc:
                import_errors.append(f"{module_path}: {type(exc).__name__}: {exc}")
        self._load_error = "failed to import LeRobot SmolVLAPolicy; tried " + "; ".join(import_errors)
        return None

    def _make_lerobot_batch(self, observation: VLAObservation) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("torch is required for native SmolVLA inference") from exc

        image = np.asarray(observation.image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        state = extract_state_features(observation.state).astype(np.float32)
        return {
            "observation.images.front": torch.from_numpy(image).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task": [observation.instruction],
        }
