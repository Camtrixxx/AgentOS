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
from agent.vision_bc_policy import VisionBCPolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy in FakeManipulationEnv.")
    parser.add_argument("--policy", default="bc", choices=["bc", "vision_bc", "scripted"])
    parser.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--randomize-layout", action="store_true")
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

    for episode_idx in range(args.num_episodes):
        task = tasks[episode_idx % len(tasks)]
        include_image = args.policy == "vision_bc"
        config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            include_image=include_image,
            randomize_layout=args.randomize_layout,
        )
        env = FakeManipulationEnv(config=config, seed=episode_idx)
        policy = make_policy(args.policy, PROJECT_ROOT / args.checkpoint)
        result = AgentLoop(env, policy).run_episode(task=task, max_steps=args.max_steps)
        successes += int(result.success)
        total_steps += result.steps
        print(
            f"episode={episode_idx} target={task.target_color} "
            f"success={result.success} steps={result.steps}"
        )

    print(f"success_rate={successes / args.num_episodes:.3f}")
    print(f"avg_steps={total_steps / args.num_episodes:.2f}")


def make_policy(policy_name: str, checkpoint_path: Path):
    if policy_name == "scripted":
        return ScriptedPickPlacePolicy()
    if policy_name == "bc":
        return BCPolicy(checkpoint_path)
    if policy_name == "vision_bc":
        return VisionBCPolicy(checkpoint_path)
    raise ValueError(f"Unknown policy {policy_name!r}")


if __name__ == "__main__":
    main()
