from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_loop import run_episode
from agent.scripted_policy import ScriptedPickPlacePolicy
from recorders.episode_recorder import EpisodeRecorder, EpisodeRecorderConfig
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a language-conditioned fake embodied agent.")
    parser.add_argument("--instruction", default="pick up the red block and place it in the bowl")
    parser.add_argument("--target-color", default="red", choices=["red", "blue", "green"])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--record", action="store_true", help="Record the rollout under data/demos")
    parser.add_argument("--output-dir", default="data/demos")
    parser.add_argument("--include-image", action="store_true", help="Include rendered RGB images in observations")
    parser.add_argument("--randomize-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        include_image=args.include_image,
        randomize_layout=args.randomize_layout,
    )
    env = FakeManipulationEnv(config=config, seed=0)
    policy = ScriptedPickPlacePolicy()
    recorder = None
    if args.record:
        recorder = EpisodeRecorder(EpisodeRecorderConfig(output_dir=PROJECT_ROOT / args.output_dir))

    task = TaskSpec(instruction=args.instruction, target_color=args.target_color)
    result = run_episode(env, policy, recorder=recorder, task=task, max_steps=args.max_steps)
    print(f"success={result.success}")
    print(f"steps={result.steps}")
    print(f"total_reward={result.total_reward:.3f}")
    if recorder is not None:
        print(f"recorded_dir={PROJECT_ROOT / args.output_dir / f'episode_{recorder.episode_index - 1:06d}'}")


if __name__ == "__main__":
    main()
