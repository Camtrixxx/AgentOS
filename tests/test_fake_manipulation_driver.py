import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.fake_manipulation_driver import FakeManipulationDriver


def test_fake_manipulation_driver_reset_and_step():
    driver = FakeManipulationDriver(seed=0)

    reset_result = driver.execute_action(
        "reset",
        {"instruction": "pick up the green block and place it in the bowl", "target_color": "green"},
    )
    step_result = driver.execute_action("env_step", {"action": [0.02, 0.0, -1.0]})
    environment = driver.get_environment()

    assert reset_result["success"]
    assert step_result["success"]
    assert environment["task"]["target_color"] == "green"
    assert environment["episode"]["step_count"] == 1


def test_fake_manipulation_driver_rejects_bad_action_shape():
    driver = FakeManipulationDriver(seed=0)

    result = driver.execute_action("env_step", {"action": [0.0, 0.0]})

    assert not result["success"]
    assert "expected action shape" in result["message"]

