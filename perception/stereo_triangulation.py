from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StereoCameraConfig:
    """Minimal stereo calibration used by the triangulator."""

    k_left: np.ndarray
    k_right: np.ndarray
    r_left_to_right: np.ndarray
    t_left_to_right: np.ndarray


class StereoHandTriangulation:
    """Triangulate paired 2D hand keypoints into a 3D hand skeleton.

    The class intentionally has no dependency on a detector. In production,
    MediaPipe, FoundationPose-style trackers, or a custom model can feed the
    2D keypoints. This keeps the perception layer easy to test.
    """

    def __init__(self, config: StereoCameraConfig, max_depth_m: float = 5.0):
        self.config = config
        self.max_depth_m = max_depth_m
        t = config.t_left_to_right.reshape(3, 1)
        self.proj_left = config.k_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.proj_right = config.k_right @ np.hstack([config.r_left_to_right, t])
        self._last_valid: Optional[np.ndarray] = None

    def triangulate(self, uv_left: np.ndarray, uv_right: np.ndarray) -> np.ndarray:
        """Return an ``(N, 3)`` array of 3D points in meters."""

        uv_left = self._validate_points(uv_left, "uv_left")
        uv_right = self._validate_points(uv_right, "uv_right")
        if uv_left.shape != uv_right.shape:
            raise ValueError(f"Point shapes differ: {uv_left.shape} vs {uv_right.shape}")

        points_4d = self._linear_triangulate(uv_left, uv_right)
        points_3d = points_4d[:, :3] / points_4d[:, 3:4]

        if np.linalg.norm(self.config.t_left_to_right) > 10.0:
            points_3d = points_3d / 1000.0

        valid = (
            np.isfinite(points_3d).all(axis=1)
            & (points_3d[:, 2] > 0.0)
            & (points_3d[:, 2] < self.max_depth_m)
        )
        points_3d = self._repair_invalid(points_3d, valid)
        self._last_valid = points_3d.copy()
        return points_3d

    def _linear_triangulate(self, uv_left: np.ndarray, uv_right: np.ndarray) -> np.ndarray:
        points = []
        for left, right in zip(uv_left, uv_right):
            u0, v0 = left
            u1, v1 = right
            a = np.vstack(
                [
                    u0 * self.proj_left[2] - self.proj_left[0],
                    v0 * self.proj_left[2] - self.proj_left[1],
                    u1 * self.proj_right[2] - self.proj_right[0],
                    v1 * self.proj_right[2] - self.proj_right[1],
                ]
            )
            _, _, vh = np.linalg.svd(a)
            points.append(vh[-1])
        return np.asarray(points, dtype=float)

    def _repair_invalid(self, points_3d: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if valid.all():
            return points_3d
        if not valid.any():
            if self._last_valid is not None and self._last_valid.shape == points_3d.shape:
                return self._last_valid.copy()
            raise ValueError("Triangulation produced no valid 3D points")

        repaired = points_3d.copy()
        fallback = repaired[valid].mean(axis=0)
        if self._last_valid is not None and self._last_valid.shape == repaired.shape:
            repaired[~valid] = self._last_valid[~valid]
        else:
            repaired[~valid] = fallback
        return repaired

    @staticmethod
    def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {points.shape}")
        return points

