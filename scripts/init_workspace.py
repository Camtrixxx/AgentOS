from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.workspace import initialize_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize embodied lab workspace Markdown files.")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = initialize_workspace(PROJECT_ROOT / args.workspace, overwrite=args.overwrite)
    print(f"workspace={paths.root}")
    print(f"action={paths.action}")
    print(f"environment={paths.environment}")
    print(f"embodied={paths.embodied}")
    print(f"plan={paths.plan}")
    print(f"report={paths.report}")


if __name__ == "__main__":
    main()
