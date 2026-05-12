import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.action_queue import append_action, load_action_document, save_action_document
from runtime.environment_io import load_environment_document
from runtime.watchdog import poll_once
from runtime.workspace import initialize_workspace


def test_watchdog_executes_first_pending_action(tmp_path):
    paths = initialize_workspace(tmp_path / "workspace")
    document = append_action(None, action_type="env_step", parameters={"action": [0.02, 0.0, -1.0]})
    save_action_document(paths.action, document)

    result = poll_once(FakeManipulationDriver(seed=0), paths)
    actions = load_action_document(paths.action)["actions"]
    environment = load_environment_document(paths.environment)

    assert result is not None
    assert result["success"]
    assert actions[0]["status"] == "completed"
    assert environment["episode"]["step_count"] == 1


def test_workspace_initialization_creates_plan_report_files(tmp_path):
    paths = initialize_workspace(tmp_path / "workspace")

    assert paths.plan.exists()
    assert paths.report.exists()
    assert paths.skill.exists()
