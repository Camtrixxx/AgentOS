from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.agent_loop import Policy, run_episode
from envs.fake_manipulation_env import FakeManipulationEnv, TaskSpec
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.planner import TaskPlan
from runtime.repository import WorkspaceRepository
from runtime.trace import TraceLogger
from tools.embodied_tools import ReadEnvironmentTool, ResetTaskTool, ScriptedPickPlaceLoopTool, StepEnvTool
from tools.evaluation_tools import EvaluateScriptedPolicyTool
from tools.registry import ToolRegistry
from tools.render_tools import RenderFakeEnvTool


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    steps: int
    last_reward: float
    render_path: str | None
    trace_path: str | None
    report_path: str
    step_records: list[dict[str, Any]] | None = None


class AgentOSExecutor:
    """Execute a TaskPlan through tools, ACTION.md, watchdog, and HAL."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        driver: FakeManipulationDriver,
        registry: ToolRegistry | None = None,
        trace: TraceLogger | None = None,
    ):
        self.repo = WorkspaceRepository(workspace)
        self.driver = driver
        self.trace = trace
        self.registry = registry or build_default_registry(
            workspace=self.repo,
            driver=driver,
            trace=trace,
        )

    def execute(
        self,
        plan: TaskPlan,
        *,
        max_steps: int,
        render_output: str | Path | None = None,
    ) -> ExecutionResult:
        self.repo.initialize()
        self.repo.save_plan(plan)
        self._log("plan_saved", {"path": str(self.repo.paths.plan)})

        step_records: list[dict[str, Any]] = []
        failed = False
        failure_text: str | None = None

        for index, step in enumerate(plan.steps, start=1):
            parameters = dict(step.parameters)
            if step.tool == "scripted_pick_place_loop":
                parameters.setdefault("max_steps", max_steps)
            if step.tool == "render_fake_env" and render_output is not None:
                parameters.setdefault("output", str(render_output))

            response = self.registry.run(step.tool, parameters)
            record = {
                "index": index,
                "tool": step.tool,
                "description": step.description,
                "parameters": parameters,
                "response": response.to_dict(),
            }
            step_records.append(record)
            self._log("plan_step_finished", record)
            if response.error is not None:
                failed = True
                failure_text = response.text
                break

        environment = self.repo.get_environment()
        episode = environment.get("episode", {}) if isinstance(environment.get("episode"), dict) else {}
        success = bool(episode.get("success", False)) and not failed
        steps = int(episode.get("step_count", 0) or 0)
        last_reward = float(episode.get("last_reward", 0.0) or 0.0)
        render_path = _find_render_path(step_records)
        if render_path is None and render_output is not None:
            render_path = str(render_output)

        report = {
            "instruction": plan.instruction,
            "target_color": plan.target_color,
            "success": success,
            "steps": steps,
            "last_reward": last_reward,
            "render_path": render_path,
            "trace_path": str(self.trace.path) if self.trace is not None else None,
            "plan_path": str(self.repo.paths.plan),
            "step_records": step_records,
            "environment": environment,
        }
        if failure_text is not None:
            report["failure"] = failure_text
        self.repo.save_report(report)
        if not success:
            _record_failure(self.repo, failure_text or f"Task ended without success after {steps} steps")
        self._log(
            "execution_report_saved",
            {"path": str(self.repo.paths.report), "success": success, "steps": steps},
        )

        return ExecutionResult(
            success=success,
            steps=steps,
            last_reward=last_reward,
            render_path=render_path,
            trace_path=str(self.trace.path) if self.trace is not None else None,
            report_path=str(self.repo.paths.report),
            step_records=step_records,
        )

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.log(event, payload)


def build_default_registry(
    *,
    workspace: str | Path | WorkspaceRepository,
    driver: FakeManipulationDriver,
    trace: TraceLogger | None = None,
    policy: Policy | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(trace_logger=trace)
    registry.register(ReadEnvironmentTool(workspace))
    registry.register(ResetTaskTool(workspace, driver=driver))
    registry.register(StepEnvTool(workspace, driver=driver))
    registry.register(ScriptedPickPlaceLoopTool(workspace, driver=driver, policy=policy))
    registry.register(RenderFakeEnvTool(workspace))
    registry.register(EvaluateScriptedPolicyTool())
    return registry


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
    driver = FakeManipulationDriver(env=env)
    registry = build_default_registry(workspace=WorkspaceRepository(workspace), driver=driver, trace=trace, policy=policy)
    executor = AgentOSExecutor(workspace=workspace, driver=driver, registry=registry, trace=trace)
    return executor.execute(plan, max_steps=max_steps, render_output=render_output)


def execute_direct_task_plan(
    plan: TaskPlan,
    *,
    env: FakeManipulationEnv,
    policy: Policy,
    workspace: str | Path,
    max_steps: int,
) -> ExecutionResult:
    """Direct env-policy rollout for offline evaluation paths."""

    repo = WorkspaceRepository(workspace)
    repo.initialize()
    repo.save_plan(plan)
    task = TaskSpec(instruction=plan.instruction, target_color=plan.target_color)
    result = run_episode(env, policy, task=task, max_steps=max_steps)
    report = {
        "instruction": plan.instruction,
        "target_color": plan.target_color,
        "success": result.success,
        "steps": result.steps,
        "last_reward": result.total_reward,
        "execution_mode": "direct_rollout",
    }
    repo.save_report(report)
    return ExecutionResult(
        success=result.success,
        steps=result.steps,
        last_reward=result.total_reward,
        render_path=None,
        trace_path=None,
        report_path=str(repo.paths.report),
        step_records=[],
    )


def _record_failure(repo: WorkspaceRepository, details: str) -> None:
    repo.append_lesson(title="Execution issue", details=details)


def _find_render_path(step_records: list[dict[str, Any]]) -> str | None:
    for record in reversed(step_records):
        response = record.get("response", {})
        data = response.get("data", {}) if isinstance(response, dict) else {}
        path = data.get("path") if isinstance(data, dict) else None
        if record.get("tool") == "render_fake_env" and path is not None:
            return str(path)
    return None
