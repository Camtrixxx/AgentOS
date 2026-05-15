from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.executor import AgentOSExecutor
from runtime.planner import Planner, RuleBasedPlanner
from runtime.trace import TraceLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the file-backed embodied AgentOS runtime.")
    parser.add_argument("instruction", nargs="?", default="pick up the red block and place it in the bowl")
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default=None)
    parser.add_argument("--workspace", default="workspace/agentos")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--render-output", default="outputs/agentos_env.ppm")
    parser.add_argument(
        "--planner",
        choices=["rule", "deepseek"],
        default="rule",
        help="Planner backend. Defaults to deterministic rule planner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = PROJECT_ROOT / args.workspace

    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        randomize_layout=args.randomize_layout,
    )
    env = FakeManipulationEnv(config=config, seed=args.seed)
    driver = FakeManipulationDriver(env=env)
    trace = TraceLogger(PROJECT_ROOT / "outputs" / "traces")
    planner: Planner
    if args.planner == "deepseek":
        from runtime.llm_planner import DeepSeekPlanner

        planner = DeepSeekPlanner()
    else:
        planner = RuleBasedPlanner()
    plan = planner.plan(args.instruction, target_color=args.target_color)

    result = AgentOSExecutor(workspace=workspace, driver=driver, trace=trace).execute(
        plan,
        max_steps=args.max_steps,
        render_output=PROJECT_ROOT / args.render_output,
    )

    print(f"success={result.success}")
    print(f"steps={result.steps}")
    print(f"rendered_image={result.render_path}")
    print(f"trace={trace.path}")
    print(f"report={result.report_path}")


if __name__ == "__main__":
    main()
