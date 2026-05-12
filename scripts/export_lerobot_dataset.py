from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lerobot_exporter import export_vision_demos_to_lerobot_jsonl
from datasets.lerobot_exporter import export_vision_demos_to_lerobot_native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export vision demos to a LeRobot-style JSONL manifest.")
    parser.add_argument("--data-dir", default="data/vision_demos_random")
    parser.add_argument("--output-dir", default="data/lerobot_fake_manipulation")
    parser.add_argument("--dataset-name", default="embodied_lab_fake_manipulation")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--format", choices=["manifest", "native"], default="manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = export_vision_demos_to_lerobot_native if args.format == "native" else export_vision_demos_to_lerobot_jsonl
    summary = exporter(
        data_dir=PROJECT_ROOT / args.data_dir,
        output_dir=PROJECT_ROOT / args.output_dir,
        dataset_name=args.dataset_name,
        copy_images=args.copy_images,
    )
    print(f"output_dir={summary.output_dir}")
    print(f"num_episodes={summary.num_episodes}")
    print(f"num_frames={summary.num_frames}")
    print(f"action_dim={summary.action_dim}")
    print(f"state_dim={summary.state_dim}")
    print(f"native={summary.native}")


if __name__ == "__main__":
    main()
