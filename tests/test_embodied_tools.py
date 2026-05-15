import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.repository import WorkspaceRepository
from tools.embodied_tools import ResetTaskTool, RunWatchdogOnceTool, StepEnvTool
from tools.evaluation_tools import EvaluateScriptedPolicyTool
from tools.render_tools import RenderFakeEnvTool


def test_render_fake_env_tool_writes_ppm(tmp_path):
    repo = WorkspaceRepository(tmp_path)
    repo.initialize()
    output = tmp_path / "fake_env_tool.ppm"

    response = RenderFakeEnvTool(tmp_path, output=output).run({})

    assert response.status.value == "success"
    assert output.exists()
    assert output.read_bytes().startswith(b"P6")
    assert response.data["path"] == str(output)


def test_step_env_tool_executes_single_action(tmp_path):
    driver = FakeManipulationDriver(seed=0)
    reset = ResetTaskTool(tmp_path, driver=driver).run(
        {
            "instruction": "pick up the red block and place it in the bowl",
            "target_color": "red",
            "receptacle_name": "bowl",
        }
    )

    response = StepEnvTool(tmp_path, driver=driver).run({"action": [0.0, 0.0, -1.0]})

    assert reset.status.value == "success"
    assert response.status.value == "success"
    assert response.data["result"]["action_type"] == "env_step"


def test_step_env_tool_requires_action(tmp_path):
    response = StepEnvTool(tmp_path).run({})

    assert response.status.value == "error"
    assert response.error["code"] == "invalid_action"


def test_evaluate_scripted_policy_tool_runs_episodes(tmp_path):
    response = EvaluateScriptedPolicyTool(report_dir=tmp_path).run(
        {"num_episodes": 2, "randomize_layout": False, "write_report": False}
    )

    assert response.status.value == "success"
    assert response.data["success_rate"] == 1.0
    assert len(response.data["episodes"]) == 2


def test_run_watchdog_once_tool_empty_queue_returns_partial(tmp_path):
    repo = WorkspaceRepository(tmp_path)
    repo.initialize()

    response = RunWatchdogOnceTool(tmp_path, driver=FakeManipulationDriver(seed=0)).run({})

    assert response.status.value == "partial"
    assert response.data["result"] is None
