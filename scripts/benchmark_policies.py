from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small policy benchmark over available baselines.")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/policy_benchmark")
    parser.add_argument("--include-vision-bc", action="store_true")
    parser.add_argument("--vision-bc-checkpoint", default="checkpoints/vision_bc_random_policy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("scripted", ["--policy", "scripted"]),
        ("mock_vla", ["--policy", "vla", "--vla-backend", "mock"]),
        ("smolvla_dry_run", ["--policy", "vla", "--vla-backend", "smolvla_dry_run"]),
        ("rl_scripted", ["--policy", "rl", "--rl-backend", "scripted"]),
        ("rl_random", ["--policy", "rl", "--rl-backend", "random"]),
    ]
    if args.include_vision_bc:
        jobs.append(
            (
                "vision_bc",
                [
                    "--policy",
                    "vision_bc",
                    "--checkpoint",
                    args.vision_bc_checkpoint,
                ],
            )
        )

    results = []
    for name, policy_args in jobs:
        report_dir = output_dir / name
        cmd = [
            sys.executable,
            "learning/evaluate_policy.py",
            *policy_args,
            "--num-episodes",
            str(args.num_episodes),
            "--max-steps",
            str(args.max_steps),
            "--write-report",
            "--report-dir",
            str(report_dir),
        ]
        print(f"running {name}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        result = {
            "name": name,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        result.update(_parse_metrics(completed.stdout))
        results.append(result)
        print(completed.stdout.strip())
        if completed.returncode != 0:
            print(completed.stderr.strip())

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(_render_markdown(results), encoding="utf-8")
    print(f"summary_json={summary_path}")
    print(f"summary_markdown={markdown_path}")


def _parse_metrics(stdout: str) -> dict[str, float]:
    metrics = {}
    for line in stdout.splitlines():
        if line.startswith("success_rate="):
            metrics["success_rate"] = float(line.split("=", 1)[1])
        elif line.startswith("avg_steps="):
            metrics["avg_steps"] = float(line.split("=", 1)[1])
        elif line.startswith("avg_reward="):
            metrics["avg_reward"] = float(line.split("=", 1)[1])
    return metrics


def _render_markdown(results: list[dict]) -> str:
    rows = [
        "| Policy | Return Code | Success Rate | Avg Steps | Avg Reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        rows.append(
            f"| {result['name']} | {result['returncode']} | "
            f"{result.get('success_rate', 0.0):.3f} | "
            f"{result.get('avg_steps', 0.0):.2f} | "
            f"{result.get('avg_reward', 0.0):.3f} |"
        )
    return "\n".join(["# Policy Benchmark", "", *rows, ""])


if __name__ == "__main__":
    main()

