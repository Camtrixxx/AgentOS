from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.watchdog import run_watchdog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embodied lab watchdog.")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Poll once and exit.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    driver = FakeManipulationDriver(
        seed=args.seed,
        include_image=args.include_image,
        randomize_layout=args.randomize_layout,
        max_steps=args.max_steps,
    )
    run_watchdog(
        driver,
        workspace=PROJECT_ROOT / args.workspace,
        poll_interval=args.poll_interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()

