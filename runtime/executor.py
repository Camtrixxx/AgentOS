from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.scripted_policy import ScriptedPickPlacePolicy
from runtime.lessons import append_lesson
from runtime.observation import observation_from_environment
from runtime.plan_io import save_execution_report, save_plan_document
from runtime.planner import TaskPlan
from runtime.trace import TraceLogger
from runtime.workspace import WorkspacePaths, initialize_workspace
from tools.registry import ToolRegistry


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    steps: int
    last_reward: float
    render_path: str | None
    trace_path: str | None
    report_path: str


def execute_task_plan(
    plan: TaskPlan,
    *,
    registry: ToolRegistry,
    workspace: str | Path,
    max_steps: int,
    render_output: str | Path,
    trace: TraceLogger | None = None,
) -> ExecutionResult:
    paths = initialize_workspace(workspace)
    save_plan_document(paths.plan, plan)
    if trace is not None:
        trace.log("plan_saved", {"path": str(paths.plan)})

    reset_step = next((step for step in plan.steps if step.tool == "reset_task"), None)
    if reset_step is None:
        _record_failure(paths, "Planner produced no reset_task step")
        raise ValueError("plan must include reset_task")

    reset_params = dict(reset_step.parameters)
    reset_params["workspace"] = paths.root
    reset_response = registry.run(reset_step.tool, reset_params)
    if reset_response.error is not None:
        _record_failure(paths, reset_response.text)
        raise RuntimeError(reset_response.text)

    policy = ScriptedPickPlacePolicy()
    success = False
    steps = 0
    last_reward = 0.0
    render_path: str | None = None
    step_records: list[dict[str, Any]] = []

    for step_idx in range(max_steps):
        environment_response = registry.run("read_environment", {"workspace": paths.root})
        if environment_response.error is not None:
            _record_failure(paths, environment_response.text)
            raise RuntimeError(environment_response.text)
        environment = environment_response.data["environment"]
        action = policy.act(observation_from_environment(environment))
        step_response = registry.run("step_env", {"workspace": paths.root, "action": action.tolist()})
        if step_response.error is not None:
            _record_failure(paths, step_response.text)
            raise RuntimeError(step_response.text)

        environment = step_response.data["environment"]
        episode = environment.get("episode", {})
        steps = int(episode.get("step_count", step_idx + 1))
        success = bool(episode.get("success", False))
        last_reward = float(episode.get("last_reward", 0.0))
        step_records.append({"step": steps, "success": success, "reward": last_reward, "action": action.tolist()})
        print(f"step={steps} success={success} reward={last_reward:.3f}")
        if bool(episode.get("done", False)):
            break

    render_response = registry.run("render_fake_env", {"workspace": paths.root, "output": render_output})
    if render_response.error is None:
        render_path = render_response.data.get("path")

    report = {
        "instruction": plan.instruction,
        "target_color": plan.target_color,
        "success": success,
        "steps": steps,
        "last_reward": last_reward,
        "render_path": render_path,
        "trace_path": str(trace.path) if trace is not None else None,
        "plan_path": str(paths.plan),
        "step_records": step_records,
    }
    save_execution_report(paths.report, report)
    if not success:
        _record_failure(paths, f"Task ended without success after {steps} steps")
    if trace is not None:
        trace.log("execution_report_saved", {"path": str(paths.report), "success": success, "steps": steps})

    return ExecutionResult(
        success=success,
        steps=steps,
        last_reward=last_reward,
        render_path=render_path,
        trace_path=str(trace.path) if trace is not None else None,
        report_path=str(paths.report),
    )


def _record_failure(paths: WorkspacePaths, details: str) -> None:
    append_lesson(paths.lessons, title="Execution issue", details=details)

