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


class AgentLoop:
    """Run policy-environment rollouts and optionally record transitions."""

    def __init__(self, env: Env, policy: Policy, recorder: Any | None = None):
        self.env = env
        self.policy = policy
        self.recorder = recorder

    def run_episode(self, task: Any | None = None, max_steps: int | None = None) -> RolloutResult:
        observation = self.env.reset(task=task) if task is not None else self.env.reset()
        if self.recorder is not None:
            self.recorder.start_episode(observation)

        total_reward = 0.0
        success = False
        steps = 0
        limit = max_steps if max_steps is not None else 10_000

        for step_idx in range(limit):
            action = self.policy.act(observation)
            next_observation, reward, done, info = self.env.step(action)
            total_reward += reward
            steps = step_idx + 1
            success = bool(info.get("success", False))

            if self.recorder is not None:
                self.recorder.record_step(observation, action, reward, next_observation, done, info)

            observation = next_observation
            if done:
                break

        if self.recorder is not None:
            self.recorder.end_episode({"steps": steps, "total_reward": total_reward, "success": success})
        return RolloutResult(steps=steps, total_reward=total_reward, success=success)

