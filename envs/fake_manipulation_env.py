from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjectState:
    name: str
    color: str
    position: np.ndarray


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    target_color: str
    receptacle_name: str = "bowl"


@dataclass(frozen=True)
class FakeManipulationConfig:
    workspace_low: np.ndarray
    workspace_high: np.ndarray
    max_steps: int = 80
    step_size: float = 0.06
    grasp_radius: float = 0.08
    place_radius: float = 0.1
    image_size: int = 128
    include_image: bool = False
    randomize_layout: bool = False
    object_position_noise: float = 0.16
    bowl_position_noise: float = 0.12
    min_object_distance: float = 0.22


class FakeManipulationEnv:
    """Small 2D pick-and-place environment for agent-loop development.

    The environment is intentionally simple. It gives the project a stable API
    for language-conditioned observation/action loops before introducing a real
    simulator such as LIBERO, robosuite, ManiSkill, or Isaac Lab.
    """

    ACTION_DIM = 3
    COLOR_RGB = {
        "red": np.array([220, 64, 64], dtype=np.uint8),
        "blue": np.array([72, 112, 220], dtype=np.uint8),
        "green": np.array([64, 172, 92], dtype=np.uint8),
    }

    def __init__(self, config: FakeManipulationConfig | None = None, seed: int = 0):
        self.config = config or FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
        )
        self.rng = np.random.default_rng(seed)
        self.task = TaskSpec("pick up the red block and place it in the bowl", "red")
        self.ee_position = np.zeros(2, dtype=float)
        self.gripper_closed = False
        self.held_object: str | None = None
        self.objects: dict[str, ObjectState] = {}
        self.receptacles: dict[str, np.ndarray] = {}
        self.step_count = 0

    def reset(self, task: TaskSpec | None = None) -> dict[str, Any]:
        self.task = task or self.task
        self.ee_position = np.array([0.0, -0.75], dtype=float)
        self.gripper_closed = False
        self.held_object = None
        self.step_count = 0

        self.objects, self.receptacles = self._make_layout()
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(f"Expected action shape ({self.ACTION_DIM},), got {action.shape}")

        delta = np.clip(action[:2], -self.config.step_size, self.config.step_size)
        self.ee_position = np.clip(
            self.ee_position + delta,
            self.config.workspace_low,
            self.config.workspace_high,
        )
        close_gripper = action[2] > 0.0
        self._update_gripper(close_gripper)
        self.step_count += 1

        success = self._is_success()
        timeout = self.step_count >= self.config.max_steps
        reward = 1.0 if success else -0.01
        done = success or timeout
        info = {"success": success, "timeout": timeout, "held_object": self.held_object}
        return self._observation(), reward, done, info

    def action_space_sample(self) -> np.ndarray:
        xy = self.rng.uniform(-self.config.step_size, self.config.step_size, size=2)
        grip = self.rng.choice([-1.0, 1.0], size=1)
        return np.concatenate([xy, grip])

    def _make_layout(self) -> tuple[dict[str, ObjectState], dict[str, np.ndarray]]:
        base_objects = {
            "red_block": ("red", np.array([-0.55, 0.15], dtype=float)),
            "blue_block": ("blue", np.array([0.45, 0.1], dtype=float)),
            "green_block": ("green", np.array([-0.1, 0.45], dtype=float)),
        }
        base_bowl = np.array([0.55, 0.65], dtype=float)
        if not self.config.randomize_layout:
            return (
                {
                    name: ObjectState(name, color, position.copy())
                    for name, (color, position) in base_objects.items()
                },
                {"bowl": base_bowl.copy()},
            )

        sampled_positions: dict[str, np.ndarray] = {}
        for name, (_, base_position) in base_objects.items():
            sampled_positions[name] = self._sample_near(
                base_position,
                self.config.object_position_noise,
                existing=list(sampled_positions.values()),
            )
        objects = {
            name: ObjectState(name, base_objects[name][0], position)
            for name, position in sampled_positions.items()
        }
        bowl = self._sample_near(base_bowl, self.config.bowl_position_noise, existing=[])
        return objects, {"bowl": bowl}

    def _sample_near(
        self,
        base_position: np.ndarray,
        noise: float,
        existing: list[np.ndarray],
        max_attempts: int = 50,
    ) -> np.ndarray:
        for _ in range(max_attempts):
            candidate = base_position + self.rng.uniform(-noise, noise, size=2)
            candidate = self._clip_to_workspace(candidate)
            if all(np.linalg.norm(candidate - other) >= self.config.min_object_distance for other in existing):
                return candidate
        return self._clip_to_workspace(base_position)

    def _clip_to_workspace(self, position: np.ndarray) -> np.ndarray:
        margin = 0.08
        return np.clip(
            np.asarray(position, dtype=float),
            self.config.workspace_low + margin,
            self.config.workspace_high - margin,
        )

    def render_rgb(self) -> np.ndarray:
        """Render the top-down fake workspace as an RGB uint8 image."""

        image = np.full(
            (self.config.image_size, self.config.image_size, 3),
            fill_value=245,
            dtype=np.uint8,
        )
        self._draw_grid(image)
        for name, pos in self.receptacles.items():
            self._draw_circle(image, pos, radius_px=13, color=np.array([246, 196, 82], dtype=np.uint8))
            self._draw_circle(image, pos, radius_px=8, color=np.array([252, 230, 160], dtype=np.uint8))

        for obj in self.objects.values():
            color = self.COLOR_RGB.get(obj.color, np.array([128, 128, 128], dtype=np.uint8))
            self._draw_square(image, obj.position, half_size_px=6, color=color)

        ee_color = np.array([30, 30, 30], dtype=np.uint8)
        if self.gripper_closed:
            ee_color = np.array([20, 20, 20], dtype=np.uint8)
        self._draw_circle(image, self.ee_position, radius_px=5, color=ee_color)
        self._draw_gripper(image)
        return image

    def _update_gripper(self, close_gripper: bool) -> None:
        if close_gripper and not self.gripper_closed:
            nearest_name = self._nearest_object_name()
            if nearest_name is not None:
                obj = self.objects[nearest_name]
                if np.linalg.norm(obj.position - self.ee_position) <= self.config.grasp_radius:
                    self.held_object = nearest_name
        elif not close_gripper and self.gripper_closed:
            self.held_object = None

        self.gripper_closed = close_gripper
        if self.held_object is not None:
            obj = self.objects[self.held_object]
            self.objects[self.held_object] = ObjectState(obj.name, obj.color, self.ee_position.copy())

    def _nearest_object_name(self) -> str | None:
        if not self.objects:
            return None
        return min(
            self.objects,
            key=lambda name: float(np.linalg.norm(self.objects[name].position - self.ee_position)),
        )

    def _target_object_name(self) -> str:
        for name, obj in self.objects.items():
            if obj.color == self.task.target_color:
                return name
        raise ValueError(f"No object with color {self.task.target_color!r}")

    def _is_success(self) -> bool:
        target = self.objects[self._target_object_name()]
        receptacle = self.receptacles[self.task.receptacle_name]
        return (
            self.held_object is None
            and not self.gripper_closed
            and np.linalg.norm(target.position - receptacle) <= self.config.place_radius
        )

    def _observation(self) -> dict[str, Any]:
        observation = {
            "instruction": self.task.instruction,
            "ee_position": self.ee_position.copy(),
            "gripper_closed": self.gripper_closed,
            "held_object": self.held_object,
            "objects": {
                name: {"color": obj.color, "position": obj.position.copy()}
                for name, obj in self.objects.items()
            },
            "receptacles": {name: pos.copy() for name, pos in self.receptacles.items()},
            "step_count": self.step_count,
        }
        if self.config.include_image:
            observation["image"] = self.render_rgb()
        return observation

    def _world_to_pixel(self, position: np.ndarray) -> tuple[int, int]:
        low = self.config.workspace_low
        high = self.config.workspace_high
        normalized = (np.asarray(position, dtype=float) - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
        x = int(round(normalized[0] * (self.config.image_size - 1)))
        y = int(round((1.0 - normalized[1]) * (self.config.image_size - 1)))
        return x, y

    def _draw_grid(self, image: np.ndarray) -> None:
        grid_color = np.array([225, 225, 225], dtype=np.uint8)
        for frac in (0.25, 0.5, 0.75):
            idx = int(round(frac * (self.config.image_size - 1)))
            image[idx : idx + 1, :, :] = grid_color
            image[:, idx : idx + 1, :] = grid_color

    def _draw_square(self, image: np.ndarray, position: np.ndarray, half_size_px: int, color: np.ndarray) -> None:
        x, y = self._world_to_pixel(position)
        x0 = max(0, x - half_size_px)
        x1 = min(self.config.image_size, x + half_size_px + 1)
        y0 = max(0, y - half_size_px)
        y1 = min(self.config.image_size, y + half_size_px + 1)
        image[y0:y1, x0:x1, :] = color

    def _draw_circle(self, image: np.ndarray, position: np.ndarray, radius_px: int, color: np.ndarray) -> None:
        x, y = self._world_to_pixel(position)
        y_grid, x_grid = np.ogrid[: self.config.image_size, : self.config.image_size]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius_px**2
        image[mask] = color

    def _draw_gripper(self, image: np.ndarray) -> None:
        x, y = self._world_to_pixel(self.ee_position)
        span = 4 if self.gripper_closed else 8
        color = np.array([35, 35, 35], dtype=np.uint8)
        y0 = max(0, y - 8)
        y1 = min(self.config.image_size, y + 9)
        left_x = max(0, x - span)
        right_x = min(self.config.image_size - 1, x + span)
        image[y0:y1, left_x : left_x + 2, :] = color
        image[y0:y1, right_x : right_x + 2, :] = color
