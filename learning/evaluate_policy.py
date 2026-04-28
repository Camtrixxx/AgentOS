from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_loop import AgentLoop
from agent.bc_policy import BCPolicy
from agent.scripted_policy import ScriptedPickPlacePolicy
from agent.vla_policy import VLAPolicy
from agent.vision_bc_policy import VisionBCPolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec
from evaluation.report import (
    EpisodeEval,
    EvaluationSummary,
    infer_failure_reason,
    write_evaluation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy in FakeManipulationEnv.")
    parser.add_argument("--policy", default="bc", choices=["bc", "vision_bc", "vla", "scripted"])
    parser.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    parser.add_argument("--vla-backend", default="mock", choices=["mock"])
    parser.add_argument("--num-episodes", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report-dir", default="outputs/eval_reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [
        TaskSpec("pick up the red block and place it in the bowl", "red"),
        TaskSpec("pick up the blue block and place it in the bowl", "blue"),
        TaskSpec("pick up the green block and place it in the bowl", "green"),
    ]
    successes = 0
    total_steps = 0
    total_reward = 0.0
    episode_results: list[EpisodeEval] = []

    for episode_idx in range(args.num_episodes):
        task = tasks[episode_idx % len(tasks)]
        include_image = args.policy in {"vision_bc", "vla"}
        config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            include_image=include_image,
            randomize_layout=args.randomize_layout,
        )
        env = FakeManipulationEnv(config=config, seed=episode_idx)
        policy = make_policy(args.policy, PROJECT_ROOT / args.checkpoint, args.vla_backend)
        result = AgentLoop(env, policy).run_episode(task=task, max_steps=args.max_steps)
        successes += int(result.success)
        total_steps += result.steps
        total_reward += result.total_reward
        failure_reason = infer_failure_reason(result.success, result.steps, args.max_steps)
        episode_results.append(
            EpisodeEval(
                episode=episode_idx,
                target_color=task.target_color,
                success=result.success,
                steps=result.steps,
                total_reward=result.total_reward,
                failure_reason=failure_reason,
            )
        )
        print(
            f"episode={episode_idx} target={task.target_color} "
            f"success={result.success} steps={result.steps} "
            f"reward={result.total_reward:.3f} failure_reason={failure_reason}"
        )

    success_rate = successes / args.num_episodes
    avg_steps = total_steps / args.num_episodes
    avg_reward = total_reward / args.num_episodes
    print(f"success_rate={success_rate:.3f}")
    print(f"avg_steps={avg_steps:.2f}")
    print(f"avg_reward={avg_reward:.3f}")

    if args.write_report:
        checkpoint = None if args.policy in {"scripted", "vla"} else str(PROJECT_ROOT / args.checkpoint)
        summary = EvaluationSummary(
            policy=args.policy,
            checkpoint=checkpoint,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            randomize_layout=args.randomize_layout,
            success_rate=success_rate,
            avg_steps=avg_steps,
            avg_reward=avg_reward,
            episodes=episode_results,
        )
        json_path, md_path = write_evaluation_report(summary, PROJECT_ROOT / args.report_dir)
        print(f"report_json={json_path}")
        print(f"report_markdown={md_path}")


def make_policy(policy_name: str, checkpoint_path: Path, vla_backend: str):
    if policy_name == "scripted":
        return ScriptedPickPlacePolicy()
    if policy_name == "bc":
        return BCPolicy(checkpoint_path)
    if policy_name == "vision_bc":
        return VisionBCPolicy(checkpoint_path)
    if policy_name == "vla":
        if vla_backend != "mock":
            raise ValueError(f"Unsupported VLA backend {vla_backend!r}")
        return VLAPolicy()
    raise ValueError(f"Unknown policy {policy_name!r}")


if __name__ == "__main__":
    main()
