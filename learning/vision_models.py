from __future__ import annotations

import torch
from torch import nn


class VisionBCPolicyNet(nn.Module):
    """Tiny CNN policy for RGB image + task color + proprio/state BC."""

    def __init__(self, task_dim: int = 3, state_dim: int = 15, action_dim: int = 3):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + task_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, task: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        state_features = self.state_encoder(state)
        return self.head(torch.cat([image_features, state_features, task], dim=-1))
