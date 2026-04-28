from __future__ import annotations

import numpy as np


class IdentityIKSolver:
    """Placeholder IK solver with the same shape as a real solver adapter."""

    def solve(self, target_joint_position: np.ndarray) -> np.ndarray:
        return np.asarray(target_joint_position, dtype=float).copy()

