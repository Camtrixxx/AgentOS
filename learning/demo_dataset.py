from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.features import extract_state_features


class DemoTransitionDataset(Dataset):
    """Load recorded JSONL episodes as supervised BC transitions."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.features: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self._load()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index]).float(),
            torch.from_numpy(self.actions[index]).float(),
        )

    def _load(self) -> None:
        episode_files = sorted(self.data_dir.glob("episode_*/transitions.jsonl"))
        if not episode_files:
            raise FileNotFoundError(f"No recorded episodes found under {self.data_dir}")

        for path in episode_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    transition = json.loads(line)
                    self.features.append(extract_state_features(transition["observation"]))
                    self.actions.append(np.asarray(transition["action"], dtype=np.float32))

        if not self.features:
            raise ValueError(f"No transitions found under {self.data_dir}")

