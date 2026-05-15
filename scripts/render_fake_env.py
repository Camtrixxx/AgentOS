from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.fake_manipulation_env import FakeManipulationEnv, TaskSpec
from envs.ppm_writer import write_ppm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the fake manipulation environment to a PPM image.")
    parser.add_argument("--output", default="outputs/fake_env.ppm")
    parser.add_argument("--target-color", default="red", choices=["red", "blue", "green"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = FakeManipulationEnv(seed=0)
    env.reset(TaskSpec(f"pick up the {args.target_color} block and place it in the bowl", args.target_color))
    image = env.render_rgb()
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ppm(output_path, image)
    print(f"rendered_image={output_path}")
    print(f"shape={image.shape} dtype={image.dtype}")


if __name__ == "__main__":
    main()
