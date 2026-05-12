from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.scripted_policy import ScriptedPickPlacePolicy
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.observation import observation_from_environment
from runtime.trace import TraceLogger
from runtime.workspace import initialize_workspace
from tools.embodied_tools import ReadEnvironmentTool, ResetTaskTool, StepEnvTool
from tools.evaluation_tools import EvaluateScriptedPolicyTool
from tools.registry import ToolRegistry
from tools.render_tools import RenderFakeEnvTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal tool-based embodied agent.")
    parser.add_argument("instruction", nargs="?", default="pick up the red block and place it in the bowl")
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default=None)
    parser.add_argument("--workspace", default="workspace/tool_agent")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--render-output", default="outputs/tool_agent_env.ppm")
    parser.add_argument("--eval-scripted", action="store_true", help="Also run scripted policy evaluation tool.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = PROJECT_ROOT / args.workspace
    initialize_workspace(workspace, overwrite=True)

    trace = TraceLogger(PROJECT_ROOT / "outputs" / "traces")
    driver = FakeManipulationDriver(
        seed=args.seed,
        include_image=False,
        randomize_layout=args.randomize_layout,
        max_steps=args.max_steps,
    )
    registry = ToolRegistry(trace_logger=trace)
    registry.register(ReadEnvironmentTool(workspace))
    registry.register(ResetTaskTool(workspace, driver=driver))
    registry.register(StepEnvTool(workspace, driver=driver))
    registry.register(RenderFakeEnvTool(workspace, PROJECT_ROOT / args.render_output))
    registry.register(EvaluateScriptedPolicyTool(PROJECT_ROOT / "outputs" / "eval_reports_tool"))

    target_color = args.target_color or infer_target_color(args.instruction)
    trace.log("agent_start", {"instruction": args.instruction, "target_color": target_color})
    reset = registry.run(
        "reset_task",
        {
            "instruction": args.instruction,
            "target_color": target_color,
            "workspace": workspace,
        },
    )
    if reset.error is not None:
        raise SystemExit(reset.text)

    policy = ScriptedPickPlacePolicy()
    success = False
    steps = 0
    last_reward = 0.0

    for step_idx in range(args.max_steps):
        environment_response = registry.run("read_environment", {"workspace": workspace})
        environment = environment_response.data["environment"]
        observation = observation_from_environment(environment)
        action = policy.act(observation)
        step_response = registry.run("step_env", {"workspace": workspace, "action": action.tolist()})
        environment = step_response.data.get("environment", environment)
        episode = environment.get("episode", {})
        steps = int(episode.get("step_count", step_idx + 1))
        success = bool(episode.get("success", False))
        last_reward = float(episode.get("last_reward", 0.0))
        print(f"step={steps} success={success} reward={last_reward:.3f}")
        if bool(episode.get("done", False)):
            break

    render = registry.run("render_fake_env", {"workspace": workspace, "output": PROJECT_ROOT / args.render_output})
    if args.eval_scripted:
        registry.run("evaluate_scripted_policy", {"num_episodes": 3, "write_report": True})

    trace.log(
        "agent_finish",
        {
            "success": success,
            "steps": steps,
            "last_reward": last_reward,
            "render_path": render.data.get("path"),
            "trace_path": str(trace.path),
        },
    )
    print(f"success={success}")
    print(f"steps={steps}")
    print(f"rendered_image={render.data.get('path')}")
    print(f"trace={trace.path}")

def infer_target_color(instruction: str) -> str:
    lowered = instruction.lower()
    for color in ("red", "blue", "green"):
        if color in lowered:
            return color
    return "red"


if __name__ == "__main__":
    main()
