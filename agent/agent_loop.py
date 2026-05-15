from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    def act(self, observation: dict[str, Any]) -> np.ndarray:
        ...


class Env(Protocol):
    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ...

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        ...


@dataclass
class RolloutResult:
    steps: int
    total_reward: float
    success: bool


def run_episode(
    env: Env,
    policy: Policy,
    *,
    recorder: Any | None = None,
    task: Any | None = None,
    max_steps: int = 10_000,
) -> RolloutResult:
    observation = env.reset(task=task) if task is not None else env.reset()
    if recorder is not None:
        recorder.start_episode(observation)

    total_reward = 0.0
    success = False
    steps = 0

    for step_idx in range(max_steps):
        action = policy.act(observation)
        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        steps = step_idx + 1
        success = bool(info.get("success", False))

        if recorder is not None:
            recorder.record_step(observation, action, reward, next_observation, done, info)

        observation = next_observation
        if done:
            break

    if recorder is not None:
        recorder.end_episode({"steps": steps, "total_reward": total_reward, "success": success})
    return RolloutResult(steps=steps, total_reward=total_reward, success=success)

