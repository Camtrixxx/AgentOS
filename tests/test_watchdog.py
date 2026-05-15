import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.action_queue import append_action
from runtime.repository import WorkspaceRepository
from runtime.watchdog import poll_once


def test_watchdog_executes_first_pending_action(tmp_path):
    repo = WorkspaceRepository(tmp_path / "workspace")
    repo.initialize()
    document = append_action(None, action_type="env_step", parameters={"action": [0.02, 0.0, -1.0]})
    repo.save_actions(document)

    result = poll_once(FakeManipulationDriver(seed=0), repo)
    actions = repo.get_actions()["actions"]
    environment = repo.get_environment()

    assert result is not None
    assert result["success"]
    assert actions[0]["status"] == "completed"
    assert environment["episode"]["step_count"] == 1


def test_workspace_initialization_creates_plan_report_files(tmp_path):
    repo = WorkspaceRepository(tmp_path / "workspace")
    repo.initialize()

    assert repo.paths.plan.exists()
    assert repo.paths.report.exists()
    assert repo.paths.skill.exists()
