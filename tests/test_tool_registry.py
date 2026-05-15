import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from tools.embodied_tools import AppendActionTool, ReadEnvironmentTool, ResetTaskTool, ScriptedPickPlaceLoopTool
from tools.planner_tools import CreatePlanTool
from tools.registry import ToolRegistry


def test_tool_registry_runs_embodied_tools(tmp_path):
    registry = ToolRegistry()
    registry.register(AppendActionTool(tmp_path))
    registry.register(ReadEnvironmentTool(tmp_path))

    queued = registry.run("append_action", {"action_type": "env_step", "parameters": {"action": [0.0, 0.0, -1.0]}})
    environment = registry.run("read_environment", {})

    assert queued.status.value == "success"
    assert queued.data["action"]["action_type"] == "env_step"
    assert environment.data["environment"]["schema_version"] == "embodied_lab.environment.v1"


def test_append_action_tool_rejects_unsafe_action(tmp_path):
    registry = ToolRegistry()
    registry.register(AppendActionTool(tmp_path))

    response = registry.run("append_action", {"action_type": "env_step", "parameters": {"action": [0.2, 0.0, -1.0]}})

    assert response.status.value == "error"
    assert response.error["code"] == "critic_rejected_action"


def test_create_plan_tool_writes_plan(tmp_path):
    registry = ToolRegistry()
    registry.register(CreatePlanTool(tmp_path))

    response = registry.run("create_plan", {"instruction": "pick up the green block and place it in the bowl"})

    assert response.status.value == "success"
    assert response.data["plan"]["target_color"] == "green"
    assert (tmp_path / "PLAN.md").exists()


def test_scripted_pick_place_loop_runs_through_watchdog(tmp_path):
    driver = FakeManipulationDriver(seed=0)
    registry = ToolRegistry()
    registry.register(ResetTaskTool(tmp_path, driver=driver))
    registry.register(ScriptedPickPlaceLoopTool(tmp_path, driver=driver))

    reset = registry.run(
        "reset_task",
        {
            "instruction": "pick up the red block and place it in the bowl",
            "target_color": "red",
            "receptacle_name": "bowl",
        },
    )
    response = registry.run("scripted_pick_place_loop", {"max_steps": 80})

    assert reset.status.value == "success"
    assert response.status.value == "success"
    assert response.data["success"]
    assert response.data["step_records"]
