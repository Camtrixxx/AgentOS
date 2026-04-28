from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.features import FEATURE_DIM, extract_state_features
from learning.models import MLPPolicy


class BCPolicy:
    """Behavior cloning policy with the same interface as scripted policies."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        action_dim = int(checkpoint.get("action_dim", 3))
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        self.model = MLPPolicy(FEATURE_DIM, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        features = extract_state_features(observation)
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            action = self.model(x).squeeze(0).cpu().numpy()
        return action.astype(float)

