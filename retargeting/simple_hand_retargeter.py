from __future__ import annotations

import numpy as np


class SimpleHandRetargeter:
    """Map a 21-point hand skeleton to a compact six-DoF hand command.

    This is deliberately small: it gives the project a runnable baseline while
    leaving room to replace it with DexRetargeting or robot-specific mappings.
    """

    FINGERTIP_IDS = np.array([4, 8, 12, 16, 20])

    def retarget(self, hand_joints_3d: np.ndarray) -> np.ndarray:
        hand_joints_3d = np.asarray(hand_joints_3d, dtype=float)
        if hand_joints_3d.shape != (21, 3):
            raise ValueError(f"Expected hand skeleton shape (21, 3), got {hand_joints_3d.shape}")

        wrist = hand_joints_3d[0]
        fingertips = hand_joints_3d[self.FINGERTIP_IDS]
        distances = np.linalg.norm(fingertips - wrist, axis=1)
        normalized_curl = 1.0 - np.clip((distances - 0.04) / 0.16, 0.0, 1.0)
        spread = np.linalg.norm(hand_joints_3d[5] - hand_joints_3d[17])
        spread_cmd = np.clip((spread - 0.04) / 0.12, 0.0, 1.0)
        return np.concatenate([normalized_curl, [spread_cmd]])

