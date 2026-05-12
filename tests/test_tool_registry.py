import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.embodied_tools import AppendActionTool, ReadEnvironmentTool
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
