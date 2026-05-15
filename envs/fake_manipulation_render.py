from __future__ import annotations

from typing import Any

import numpy as np


COLOR_RGB = {
    "red": np.array([220, 64, 64], dtype=np.uint8),
    "blue": np.array([72, 112, 220], dtype=np.uint8),
    "green": np.array([64, 172, 92], dtype=np.uint8),
}


def render_fake_manipulation_scene(
    *,
    image_size: int,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
    objects: dict[str, Any],
    receptacles: dict[str, np.ndarray],
    ee_position: np.ndarray,
    gripper_closed: bool,
) -> np.ndarray:
    """Render a top-down RGB view for the fake manipulation scene."""

    image = np.full((image_size, image_size, 3), fill_value=245, dtype=np.uint8)
    _draw_grid(image, image_size)

    for pos in receptacles.values():
        _draw_circle(image, pos, radius_px=13, color=np.array([246, 196, 82], dtype=np.uint8), workspace_low=workspace_low, workspace_high=workspace_high)
        _draw_circle(image, pos, radius_px=8, color=np.array([252, 230, 160], dtype=np.uint8), workspace_low=workspace_low, workspace_high=workspace_high)

    for obj in objects.values():
        color = COLOR_RGB.get(obj.color, np.array([128, 128, 128], dtype=np.uint8))
        _draw_square(image, obj.position, half_size_px=6, color=color, workspace_low=workspace_low, workspace_high=workspace_high)

    ee_color = np.array([20, 20, 20], dtype=np.uint8) if gripper_closed else np.array([30, 30, 30], dtype=np.uint8)
    _draw_circle(image, ee_position, radius_px=5, color=ee_color, workspace_low=workspace_low, workspace_high=workspace_high)
    _draw_gripper(
        image,
        ee_position,
        gripper_closed=gripper_closed,
        workspace_low=workspace_low,
        workspace_high=workspace_high,
    )
    return image


def _world_to_pixel(
    position: np.ndarray,
    *,
    image_size: int,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
) -> tuple[int, int]:
    normalized = (np.asarray(position, dtype=float) - workspace_low) / (workspace_high - workspace_low)
    normalized = np.clip(normalized, 0.0, 1.0)
    x = int(round(normalized[0] * (image_size - 1)))
    y = int(round((1.0 - normalized[1]) * (image_size - 1)))
    return x, y


def _draw_grid(image: np.ndarray, image_size: int) -> None:
    grid_color = np.array([225, 225, 225], dtype=np.uint8)
    for frac in (0.25, 0.5, 0.75):
        idx = int(round(frac * (image_size - 1)))
        image[idx : idx + 1, :, :] = grid_color
        image[:, idx : idx + 1, :] = grid_color


def _draw_square(
    image: np.ndarray,
    position: np.ndarray,
    *,
    half_size_px: int,
    color: np.ndarray,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
) -> None:
    x, y = _world_to_pixel(position, image_size=image.shape[0], workspace_low=workspace_low, workspace_high=workspace_high)
    x0 = max(0, x - half_size_px)
    x1 = min(image.shape[0], x + half_size_px + 1)
    y0 = max(0, y - half_size_px)
    y1 = min(image.shape[0], y + half_size_px + 1)
    image[y0:y1, x0:x1, :] = color


def _draw_circle(
    image: np.ndarray,
    position: np.ndarray,
    *,
    radius_px: int,
    color: np.ndarray,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
) -> None:
    x, y = _world_to_pixel(position, image_size=image.shape[0], workspace_low=workspace_low, workspace_high=workspace_high)
    y_grid, x_grid = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius_px**2
    image[mask] = color


def _draw_gripper(
    image: np.ndarray,
    position: np.ndarray,
    *,
    gripper_closed: bool,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
) -> None:
    x, y = _world_to_pixel(position, image_size=image.shape[0], workspace_low=workspace_low, workspace_high=workspace_high)
    span = 4 if gripper_closed else 8
    color = np.array([35, 35, 35], dtype=np.uint8)
    y0 = max(0, y - 8)
    y1 = min(image.shape[0], y + 9)
    left_x = max(0, x - span)
    right_x = min(image.shape[1] - 1, x + span)
    image[y0:y1, left_x : left_x + 2, :] = color
    image[y0:y1, right_x : right_x + 2, :] = color
