from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.features import COLORS, extract_state_features, parse_target_color


class VisionDemoTransitionDataset(Dataset):
    """Load RGB-image demonstration transitions for vision BC."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
        self._load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, task_one_hot, state_features, action = self.samples[index]
        image = np.load(image_path).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(task_one_hot).float(),
            torch.from_numpy(state_features).float(),
            torch.from_numpy(action).float(),
        )

    def _load(self) -> None:
        episode_files = sorted(self.data_dir.glob("episode_*/transitions.jsonl"))
        if not episode_files:
            raise FileNotFoundError(f"No vision episodes found under {self.data_dir}")

        for transitions_path in episode_files:
            episode_dir = transitions_path.parent
            with transitions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    transition = json.loads(line)
                    instruction = transition["observation"]["instruction"]
                    color = parse_target_color(instruction)
                    task_one_hot = np.asarray([float(c == color) for c in COLORS], dtype=np.float32)
                    state_features = extract_state_features(transition["observation"])
                    image_path = episode_dir / transition["image_path"]
                    action = np.asarray(transition["action"], dtype=np.float32)
                    self.samples.append((image_path, task_one_hot, state_features, action))

        if not self.samples:
            raise ValueError(f"No vision transitions found under {self.data_dir}")
