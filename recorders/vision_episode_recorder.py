from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisionEpisodeRecorderConfig:
    output_dir: Path
    prefix: str = "episode"


class VisionEpisodeRecorder:
    """Record rollouts with RGB observations stored as separate NPY files."""

    def __init__(self, config: VisionEpisodeRecorderConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = self._next_index()
        self.transitions: list[dict[str, Any]] = []
        self.initial_observation: dict[str, Any] | None = None
        self._episode_dir: Path | None = None
        self._images_dir: Path | None = None

    def start_episode(self, initial_observation: dict[str, Any]) -> None:
        self.transitions = []
        self._episode_dir = self.config.output_dir / f"{self.config.prefix}_{self.episode_index:06d}"
        self._images_dir = self._episode_dir / "images"
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self._images_dir.mkdir(parents=True, exist_ok=False)
        self.initial_observation = self._strip_image(initial_observation)

    def record_step(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if self._images_dir is None:
            raise RuntimeError("start_episode must be called before record_step")
        step_idx = len(self.transitions)
        image_path = self._save_image(observation["image"], f"{step_idx:06d}.npy")
        next_image_path = self._save_image(next_observation["image"], f"{step_idx:06d}_next.npy")
        self.transitions.append(
            {
                "observation": self._strip_image(observation),
                "image_path": image_path,
                "action": np.asarray(action, dtype=float).tolist(),
                "reward": float(reward),
                "next_observation": self._strip_image(next_observation),
                "next_image_path": next_image_path,
                "done": bool(done),
                "info": self._to_jsonable(info),
            }
        )

    def end_episode(self, summary: dict[str, Any]) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("start_episode must be called before end_episode")
        metadata = {
            "initial_observation": self.initial_observation,
            "summary": self._to_jsonable(summary),
            "num_steps": len(self.transitions),
            "image_format": "npy_uint8_rgb",
        }
        (self._episode_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        with (self._episode_dir / "transitions.jsonl").open("w", encoding="utf-8") as f:
            for transition in self.transitions:
                f.write(json.dumps(transition) + "\n")

        actions = np.asarray([t["action"] for t in self.transitions], dtype=float)
        rewards = np.asarray([t["reward"] for t in self.transitions], dtype=float)
        dones = np.asarray([t["done"] for t in self.transitions], dtype=bool)
        np.savez(self._episode_dir / "arrays.npz", actions=actions, rewards=rewards, dones=dones)

        episode_dir = self._episode_dir
        self.episode_index += 1
        self._episode_dir = None
        self._images_dir = None
        return episode_dir

    def _save_image(self, image: np.ndarray, filename: str) -> str:
        if self._images_dir is None:
            raise RuntimeError("No images directory is active")
        image = np.asarray(image, dtype=np.uint8)
        path = self._images_dir / filename
        np.save(path, image)
        return str(Path("images") / filename)

    def _next_index(self) -> int:
        existing = sorted(self.config.output_dir.glob(f"{self.config.prefix}_*"))
        if not existing:
            return 0
        last = existing[-1].name.rsplit("_", maxsplit=1)[-1]
        return int(last) + 1 if last.isdigit() else len(existing)

    def _strip_image(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {key: self._to_jsonable(value) for key, value in observation.items() if key != "image"}

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

