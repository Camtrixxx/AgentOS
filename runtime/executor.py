from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.agent_loop import Policy, run_episode
from envs.fake_manipulation_env import TaskSpec
from runtime.lessons import append_lesson
from runtime.plan_io import save_execution_report, save_plan_document
from runtime.planner import TaskPlan
from runtime.trace import TraceLogger
from runtime.workspace import WorkspacePaths, initialize_workspace
from envs.ppm_writer import write_ppm


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
    env: FakeManipulationEnv,
    policy: Policy,
    workspace: str | Path,
    max_steps: int,
    render_output: str | Path | None = None,
    trace: TraceLogger | None = None,
) -> ExecutionResult:
    paths = initialize_workspace(workspace)
    save_plan_document(paths.plan, plan)
    if trace is not None:
        trace.log("plan_saved", {"path": str(paths.plan)})

    task = TaskSpec(instruction=plan.instruction, target_color=plan.target_color)
    result = run_episode(env, policy, task=task, max_steps=max_steps)

    render_path: str | None = None
    if render_output is not None:
        render_output = Path(render_output)
        image = env.render_rgb()
        render_output.parent.mkdir(parents=True, exist_ok=True)
        write_ppm(render_output, image)
        render_path = str(render_output)

    report = {
        "instruction": plan.instruction,
        "target_color": plan.target_color,
        "success": result.success,
        "steps": result.steps,
        "last_reward": result.total_reward,
        "render_path": render_path,
        "trace_path": str(trace.path) if trace is not None else None,
        "plan_path": str(paths.plan),
        "step_records": [],
    }
    save_execution_report(paths.report, report)
    if not result.success:
        _record_failure(paths, f"Task ended without success after {result.steps} steps")
    if trace is not None:
        trace.log("execution_report_saved", {"path": str(paths.report), "success": result.success, "steps": result.steps})

    return ExecutionResult(
        success=result.success,
        steps=result.steps,
        last_reward=result.total_reward,
        render_path=render_path,
        trace_path=str(trace.path) if trace is not None else None,
        report_path=str(paths.report),
    )


def _record_failure(paths: WorkspacePaths, details: str) -> None:
    append_lesson(paths.lessons, title="Execution issue", details=details)
