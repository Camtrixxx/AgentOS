from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.agent_loop import Policy, run_episode
from envs.fake_manipulation_env import FakeManipulationEnv, TaskSpec
from envs.ppm_writer import write_ppm
from runtime.planner import TaskPlan
from runtime.repository import WorkspaceRepository
from runtime.trace import TraceLogger


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
    repo = WorkspaceRepository(workspace)
    repo.initialize()
    repo.save_plan(plan)
    if trace is not None:
        trace.log("plan_saved", {"path": str(repo.paths.plan)})

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
        "plan_path": str(repo.paths.plan),
        "step_records": [],
    }
    repo.save_report(report)
    if not result.success:
        _record_failure(repo, f"Task ended without success after {result.steps} steps")
    if trace is not None:
        trace.log(
            "execution_report_saved",
            {"path": str(repo.paths.report), "success": result.success, "steps": result.steps},
        )

    return ExecutionResult(
        success=result.success,
        steps=result.steps,
        last_reward=result.total_reward,
        render_path=render_path,
        trace_path=str(trace.path) if trace is not None else None,
        report_path=str(repo.paths.report),
    )


def _record_failure(repo: WorkspaceRepository, details: str) -> None:
    repo.append_lesson(title="Execution issue", details=details)
