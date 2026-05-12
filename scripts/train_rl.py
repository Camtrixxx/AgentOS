from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gym_fake_manipulation import FakeManipulationGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL policy with optional stable-baselines3.")
    parser.add_argument("--backend", default="sb3", choices=["sb3", "smoke"])
    parser.add_argument("--output", default="checkpoints/rl_ppo_fake_manipulation.zip")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--randomize-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = FakeManipulationGymEnv(randomize_layout=args.randomize_layout)
    if args.backend == "smoke":
        obs, info = env.reset()
        action = env.env.action_space_sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"observation_dim={obs.shape[0]}")
        print(f"action_dim={action.shape[0]}")
        print(f"reward={reward:.3f} terminated={terminated} truncated={truncated}")
        return

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit("stable-baselines3 is not installed. Use --backend smoke or install stable-baselines3.") from exc

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()

