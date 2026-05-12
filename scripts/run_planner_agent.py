from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.executor import execute_task_plan
from runtime.planner import RuleBasedPlanner
from runtime.trace import TraceLogger
from runtime.workspace import initialize_workspace
from tools.embodied_tools import ReadEnvironmentTool, ResetTaskTool, StepEnvTool
from tools.planner_tools import CreatePlanTool
from tools.registry import ToolRegistry
from tools.render_tools import RenderFakeEnvTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a rule-planned, critic-checked embodied agent.")
    parser.add_argument("instruction", nargs="?", default="pick up the red block and place it in the bowl")
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default=None)
    parser.add_argument("--workspace", default="workspace/planner_agent")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--render-output", default="outputs/planner_agent_env.ppm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = PROJECT_ROOT / args.workspace
    initialize_workspace(workspace, overwrite=True)

    trace = TraceLogger(PROJECT_ROOT / "outputs" / "traces")
    planner = RuleBasedPlanner()
    plan = planner.plan(args.instruction, target_color=args.target_color)
    trace.log(
        "plan_created",
        {
            "instruction": plan.instruction,
            "target_color": plan.target_color,
            "steps": [step.__dict__ for step in plan.steps],
        },
    )

    driver = FakeManipulationDriver(
        seed=args.seed,
        include_image=False,
        randomize_layout=args.randomize_layout,
        max_steps=args.max_steps,
    )
    registry = ToolRegistry(trace_logger=trace)
    registry.register(CreatePlanTool(workspace))
    registry.register(ReadEnvironmentTool(workspace))
    registry.register(ResetTaskTool(workspace, driver=driver))
    registry.register(StepEnvTool(workspace, driver=driver))
    registry.register(RenderFakeEnvTool(workspace, PROJECT_ROOT / args.render_output))

    print("plan:")
    for index, step in enumerate(plan.steps, start=1):
        print(f"  {index}. {step.tool}: {step.description}")

    result = execute_task_plan(
        plan,
        registry=registry,
        workspace=workspace,
        max_steps=args.max_steps,
        render_output=PROJECT_ROOT / args.render_output,
        trace=trace,
    )
    trace.log(
        "agent_finish",
        {
            "success": result.success,
            "steps": result.steps,
            "last_reward": result.last_reward,
            "render_path": result.render_path,
            "trace_path": str(trace.path),
            "report_path": result.report_path,
        },
    )
    print(f"success={result.success}")
    print(f"steps={result.steps}")
    print(f"rendered_image={result.render_path}")
    print(f"trace={trace.path}")
    print(f"report={result.report_path}")


if __name__ == "__main__":
    main()
