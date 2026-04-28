from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeRecorderConfig:
    output_dir: Path
    prefix: str = "episode"


class EpisodeRecorder:
    """Record policy rollouts into simple JSONL + NPZ artifacts."""

    def __init__(self, config: EpisodeRecorderConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = self._next_index()
        self.transitions: list[dict[str, Any]] = []
        self.initial_observation: dict[str, Any] | None = None
        self.summary: dict[str, Any] = {}

    def start_episode(self, initial_observation: dict[str, Any]) -> None:
        self.transitions = []
        self.summary = {}
        self.initial_observation = self._to_jsonable(initial_observation)

    def record_step(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self.transitions.append(
            {
                "observation": self._to_jsonable(observation),
                "action": np.asarray(action, dtype=float).tolist(),
                "reward": float(reward),
                "next_observation": self._to_jsonable(next_observation),
                "done": bool(done),
                "info": self._to_jsonable(info),
            }
        )

    def end_episode(self, summary: dict[str, Any]) -> Path:
        self.summary = self._to_jsonable(summary)
        episode_dir = self.config.output_dir / f"{self.config.prefix}_{self.episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "initial_observation": self.initial_observation,
            "summary": self.summary,
            "num_steps": len(self.transitions),
        }
        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with (episode_dir / "transitions.jsonl").open("w", encoding="utf-8") as f:
            for transition in self.transitions:
                f.write(json.dumps(transition) + "\n")

        actions = np.asarray([t["action"] for t in self.transitions], dtype=float)
        rewards = np.asarray([t["reward"] for t in self.transitions], dtype=float)
        dones = np.asarray([t["done"] for t in self.transitions], dtype=bool)
        np.savez(episode_dir / "arrays.npz", actions=actions, rewards=rewards, dones=dones)

        self.episode_index += 1
        return episode_dir

    def _next_index(self) -> int:
        existing = sorted(self.config.output_dir.glob(f"{self.config.prefix}_*"))
        if not existing:
            return 0
        last = existing[-1].name.rsplit("_", maxsplit=1)[-1]
        return int(last) + 1 if last.isdigit() else len(existing)

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

