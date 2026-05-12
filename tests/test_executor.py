import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.executor import execute_task_plan
from runtime.planner import RuleBasedPlanner
from tools.embodied_tools import ReadEnvironmentTool, ResetTaskTool, StepEnvTool
from tools.registry import ToolRegistry
from tools.render_tools import RenderFakeEnvTool


def test_execute_task_plan_writes_plan_and_report(tmp_path):
    workspace = tmp_path / "workspace"
    driver = FakeManipulationDriver(seed=0)
    registry = ToolRegistry()
    registry.register(ReadEnvironmentTool(workspace))
    registry.register(ResetTaskTool(workspace, driver=driver))
    registry.register(StepEnvTool(workspace, driver=driver))
    registry.register(RenderFakeEnvTool(workspace, tmp_path / "render.ppm"))
    plan = RuleBasedPlanner().plan("pick up the red block and place it in the bowl")

    result = execute_task_plan(
        plan,
        registry=registry,
        workspace=workspace,
        max_steps=80,
        render_output=tmp_path / "render.ppm",
    )

    assert result.success
    assert result.steps < 80
    assert (workspace / "PLAN.md").exists()
    assert (workspace / "REPORT.md").exists()
    assert (tmp_path / "render.ppm").exists()

