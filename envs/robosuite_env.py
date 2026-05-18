from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RobosuiteEnvConfig:
    task_name: str = "Lift"
    robot: str = "Panda"
    horizon: int = 200
    has_renderer: bool = False
    has_offscreen_renderer: bool = False
    use_camera_obs: bool = False
    camera_name: str = "frontview"
    image_size: int = 256


class RobosuiteEnvAdapter:
    """Small adapter that normalizes robosuite into the AgentOS env shape."""

    def __init__(self, config: RobosuiteEnvConfig | None = None, *, seed: int = 0):
        self.config = config or RobosuiteEnvConfig()
        self.seed = seed
        self._robosuite = _import_robosuite()
        self.env = self._make_env()
        self.last_observation: dict[str, Any] = {}
        self.last_raw_observation: dict[str, Any] = {}
        self.step_count = 0

    @property
    def action_dim(self) -> int:
        return int(getattr(self.env, "action_dim", 0) or 0)

    def reset(self, *, instruction: str, target_color: str, receptacle_name: str = "bin") -> dict[str, Any]:
        raw = self.env.reset()
        self.step_count = 0
        return self._normalize_observation(
            raw,
            instruction=instruction,
            target_color=target_color,
            receptacle_name=receptacle_name,
        )

    def step(self, action: list[float] | np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        native_action = self.to_native_action(action)
        raw, reward, done, info = self.env.step(native_action)
        self.step_count += 1
        normalized = self._normalize_observation(raw)
        info = dict(info) if isinstance(info, dict) else {}
        info.setdefault("success", self._check_success())
        return normalized, float(reward), bool(done), info

    def to_native_action(self, action: list[float] | np.ndarray) -> np.ndarray:
        raw = np.asarray(action, dtype=float)
        native_dim = self.action_dim
        if native_dim <= 0:
            return raw
        if raw.shape == (native_dim,):
            return raw
        native = np.zeros(native_dim, dtype=float)
        if raw.shape == (3,):
            native[:2] = raw[:2]
            native[-1] = raw[2]
            return native
        if raw.shape == (4,):
            native[: min(3, native_dim)] = raw[: min(3, native_dim)]
            native[-1] = raw[3]
            return native
        raise ValueError(f"expected action shape (3,), (4,), or ({native_dim},), got {tuple(raw.shape)}")

    def render_rgb(self) -> np.ndarray:
        if self.config.has_offscreen_renderer and hasattr(self.env, "sim"):
            try:
                image = self.env.sim.render(
                    camera_name=self.config.camera_name,
                    width=self.config.image_size,
                    height=self.config.image_size,
                    depth=False,
                )
                return np.asarray(image, dtype=np.uint8)[::-1]
            except Exception:
                pass
        image_key = f"{self.config.camera_name}_image"
        image = self.last_raw_observation.get(image_key)
        if image is None:
            for key, value in self.last_raw_observation.items():
                if key.endswith("_image"):
                    image = value
                    break
        if image is not None:
            return np.asarray(image, dtype=np.uint8)
        if hasattr(self.env, "sim"):
            image = self.env.sim.render(
                camera_name=self.config.camera_name,
                width=self.config.image_size,
                height=self.config.image_size,
                depth=False,
            )
            return np.asarray(image, dtype=np.uint8)[::-1]
        image = self.env.render()
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _make_env(self):
        return self._robosuite.make(
            self.config.task_name,
            robots=self.config.robot,
            has_renderer=self.config.has_renderer,
            has_offscreen_renderer=self.config.has_offscreen_renderer,
            use_camera_obs=self.config.use_camera_obs,
            camera_names=self.config.camera_name,
            camera_heights=self.config.image_size,
            camera_widths=self.config.image_size,
            horizon=self.config.horizon,
        )

    def _normalize_observation(
        self,
        raw: dict[str, Any],
        *,
        instruction: str | None = None,
        target_color: str | None = None,
        receptacle_name: str | None = None,
    ) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        self.last_raw_observation = raw
        previous = self.last_observation
        obs = {
            "instruction": instruction or previous.get("instruction", ""),
            "ee_position": _vector(raw.get("robot0_eef_pos"), default=[0.0, 0.0, 0.0]),
            "gripper_closed": _infer_gripper_closed(raw),
            "held_object": None,
            "objects": _extract_objects(raw),
            "receptacles": {},
            "step_count": self.step_count,
            "target_color": target_color or previous.get("target_color", "red"),
            "receptacle_name": receptacle_name or previous.get("receptacle_name", "bin"),
        }
        self.last_observation = obs
        return obs

    def _check_success(self) -> bool:
        checker = getattr(self.env, "_check_success", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False


def _import_robosuite():
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        import robosuite  # type: ignore
    except ImportError as exc:
        raise ImportError("robosuite is not installed. Install with: pip install mujoco robosuite") from exc
    return robosuite


def _vector(value: Any, *, default: list[float]) -> list[float]:
    if value is None:
        return default
    array = np.asarray(value, dtype=float).reshape(-1)
    return array.tolist()


def _infer_gripper_closed(raw: dict[str, Any]) -> bool:
    qpos = raw.get("robot0_gripper_qpos")
    if qpos is None:
        return False
    array = np.asarray(qpos, dtype=float).reshape(-1)
    return bool(array.size and float(np.mean(array)) < 0.0)


def _extract_objects(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not key.endswith("_pos") or key == "robot0_eef_pos":
            continue
        if key.startswith("robot0_") or key.startswith("gripper_to_"):
            continue
        name = key[: -len("_pos")]
        objects[name] = {
            "name": name,
            "color": "unknown",
            "position": _vector(value, default=[0.0, 0.0, 0.0]),
        }
    return objects
