from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agent.agent_loop import AgentLoop
from agent.scripted_policy import ScriptedPickPlacePolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec
from evaluation.report import EpisodeEval, EvaluationSummary, infer_failure_reason, write_evaluation_report
from tools.response import ToolResponse


class EvaluateScriptedPolicyTool:
    name = "evaluate_scripted_policy"
    description = "Evaluate ScriptedPickPlacePolicy in FakeManipulationEnv and optionally write a report."

    def __init__(self, report_dir: str | Path = "outputs/eval_reports_tool"):
        self.report_dir = Path(report_dir)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        num_episodes = int(parameters.get("num_episodes", 3))
        max_steps = int(parameters.get("max_steps", 80))
        randomize_layout = bool(parameters.get("randomize_layout", False))
        write_report = bool(parameters.get("write_report", True))
        report_dir = Path(parameters.get("report_dir") or self.report_dir)

        tasks = [
            TaskSpec("pick up the red block and place it in the bowl", "red"),
            TaskSpec("pick up the blue block and place it in the bowl", "blue"),
            TaskSpec("pick up the green block and place it in the bowl", "green"),
        ]
        episodes: list[EpisodeEval] = []
        successes = 0
        total_steps = 0
        total_reward = 0.0

        for episode_idx in range(num_episodes):
            task = tasks[episode_idx % len(tasks)]
            config = FakeManipulationConfig(
                workspace_low=np.array([-1.0, -1.0], dtype=float),
                workspace_high=np.array([1.0, 1.0], dtype=float),
                randomize_layout=randomize_layout,
            )
            env = FakeManipulationEnv(config=config, seed=episode_idx)
            result = AgentLoop(env, ScriptedPickPlacePolicy()).run_episode(task=task, max_steps=max_steps)
            successes += int(result.success)
            total_steps += result.steps
            total_reward += result.total_reward
            episodes.append(
                EpisodeEval(
                    episode=episode_idx,
                    target_color=task.target_color,
                    success=result.success,
                    steps=result.steps,
                    total_reward=result.total_reward,
                    failure_reason=infer_failure_reason(result.success, result.steps, max_steps),
                )
            )

        summary = EvaluationSummary(
            policy="scripted",
            checkpoint=None,
            num_episodes=num_episodes,
            max_steps=max_steps,
            randomize_layout=randomize_layout,
            success_rate=successes / max(num_episodes, 1),
            avg_steps=total_steps / max(num_episodes, 1),
            avg_reward=total_reward / max(num_episodes, 1),
            episodes=episodes,
        )
        data: dict[str, Any] = {
            "success_rate": summary.success_rate,
            "avg_steps": summary.avg_steps,
            "avg_reward": summary.avg_reward,
            "episodes": [episode.__dict__ for episode in episodes],
        }
        if write_report:
            json_path, md_path = write_evaluation_report(summary, report_dir)
            data["report_json"] = str(json_path)
            data["report_markdown"] = str(md_path)
        return ToolResponse.success(
            f"scripted policy success_rate={summary.success_rate:.3f}",
            data=data,
        )

