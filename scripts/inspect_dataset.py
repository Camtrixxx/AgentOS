from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recorders.inspector import inspect_demo_dataset, write_inspection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect demo dataset quality.")
    parser.add_argument("--data-dir", default="data/vision_demos_random")
    parser.add_argument("--expect-images", action="store_true")
    parser.add_argument("--output-dir", default="outputs/dataset_quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_demo_dataset(PROJECT_ROOT / args.data_dir, expect_images=args.expect_images)
    json_path, md_path = write_inspection_report(inspection, PROJECT_ROOT / args.output_dir)
    print(f"ok={inspection.ok}")
    print(f"num_episodes={inspection.num_episodes}")
    print(f"num_transitions={inspection.num_transitions}")
    print(f"success_rate={inspection.success_rate:.3f}")
    print(f"target_color_counts={inspection.target_color_counts}")
    print(f"report_json={json_path}")
    print(f"report_markdown={md_path}")


if __name__ == "__main__":
    main()

