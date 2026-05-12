from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.devices import resolve_torch_device
from learning.features import COLORS, FEATURE_DIM, extract_state_features, parse_target_color
from learning.vision_models import VisionBCPolicyNet


class VisionBCPolicy:
    """CNN behavior cloning policy using observation image + language color."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = resolve_torch_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = VisionBCPolicyNet(
            task_dim=int(checkpoint.get("task_dim", len(COLORS))),
            state_dim=int(checkpoint.get("state_dim", FEATURE_DIM)),
            action_dim=int(checkpoint.get("action_dim", 3)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if "image" not in observation:
            raise ValueError("VisionBCPolicy requires observation['image']")
        image = np.asarray(observation["image"], dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        color = parse_target_color(observation["instruction"])
        task = np.asarray([float(c == color) for c in COLORS], dtype=np.float32)
        state = extract_state_features(observation)
        with torch.no_grad():
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
            task_tensor = torch.from_numpy(task).float().unsqueeze(0).to(self.device)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.model(image_tensor, task_tensor, state_tensor).squeeze(0).cpu().numpy()
        return action.astype(float)
