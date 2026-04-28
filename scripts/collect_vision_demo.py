from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_loop import AgentLoop
from agent.scripted_policy import ScriptedPickPlacePolicy
from datasets.vision_episode_recorder import VisionEpisodeRecorder, VisionEpisodeRecorderConfig
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RGB-image demonstrations in FakeManipulationEnv.")
    parser.add_argument("--num-episodes", type=int, default=9)
    parser.add_argument("--output-dir", default="data/vision_demos")
    parser.add_argument("--max-steps", type=int, default=100)
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
    for episode_idx in range(args.num_episodes):
        task = tasks[episode_idx % len(tasks)]
        config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            include_image=True,
            randomize_layout=args.randomize_layout,
        )
        env = FakeManipulationEnv(config=config, seed=episode_idx)
        policy = ScriptedPickPlacePolicy()
        recorder = VisionEpisodeRecorder(
            VisionEpisodeRecorderConfig(output_dir=PROJECT_ROOT / args.output_dir)
        )
        result = AgentLoop(env, policy, recorder=recorder).run_episode(task=task, max_steps=args.max_steps)
        successes += int(result.success)
        print(
            f"episode={episode_idx} target={task.target_color} "
            f"success={result.success} steps={result.steps}"
        )
    print(f"success_rate={successes / args.num_episodes:.3f}")


if __name__ == "__main__":
    main()
