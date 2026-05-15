from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.robosuite_env import RobosuiteEnvAdapter, RobosuiteEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify robosuite / MuJoCo reset, step, and optional render.")
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--render-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        env = RobosuiteEnvAdapter(
            RobosuiteEnvConfig(
                task_name=args.task,
                robot=args.robot,
                has_offscreen_renderer=args.offscreen or bool(args.render_output),
            )
        )
    except ImportError as exc:
        print(str(exc))
        return 2

    obs = env.reset(instruction=f"run robosuite {args.task}", target_color="red")
    print("robosuite_reset=ok")
    print(f"task={args.task}")
    print(f"robot={args.robot}")
    print(f"native_action_dim={env.action_dim}")
    print(f"normalized_keys={sorted(obs.keys())}")
    print(f"raw_keys={sorted(env.last_raw_observation.keys())[:20]}")

    action = np.zeros(env.action_dim or 4, dtype=float)
    for _ in range(args.steps):
        obs, reward, done, info = env.step(action)
    print("robosuite_step=ok")
    print(f"step_count={obs['step_count']}")
    print(f"reward={reward:.4f}")
    print(f"done={done}")
    print(f"success={bool(info.get('success', False))}")

    if args.render_output:
        from envs.ppm_writer import write_ppm

        image = env.render_rgb()
        output = PROJECT_ROOT / args.render_output
        output.parent.mkdir(parents=True, exist_ok=True)
        write_ppm(output, image)
        print(f"rendered_image={output}")
        print(f"image_shape={tuple(image.shape)}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
