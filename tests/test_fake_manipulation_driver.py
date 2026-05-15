import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from hal.base_driver import DriverState, DriverStateError
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
    assert driver.state == DriverState.IDLE
    assert environment["task"]["target_color"] == "green"
    assert environment["episode"]["step_count"] == 1


def test_fake_manipulation_driver_rejects_bad_action_shape():
    driver = FakeManipulationDriver(seed=0)

    result = driver.execute_action("env_step", {"action": [0.0, 0.0]})

    assert not result["success"]
    assert "expected action shape" in result["message"]
    assert driver.state == DriverState.FAULT


def test_driver_state_machine_blocks_actions_after_fault():
    driver = FakeManipulationDriver(seed=0)
    driver.execute_action("env_step", {"action": [0.0, 0.0]})

    with pytest.raises(DriverStateError):
        driver.execute_action("env_step", {"action": [0.0, 0.0, -1.0]})

    driver.reset_fault()
    result = driver.execute_action("env_step", {"action": [0.0, 0.0, -1.0]})

    assert result["success"]
    assert driver.state == DriverState.IDLE


def test_driver_runtime_state_reports_state():
    driver = FakeManipulationDriver(seed=0)

    runtime_state = driver.get_runtime_state()

    assert runtime_state["driver_state"] == "idle"


def test_load_environment_does_not_reset_empty_workspace_document():
    driver = FakeManipulationDriver(seed=0, randomize_layout=True)
    before_objects = {
        name: payload["position"].copy()
        for name, payload in driver.last_observation["objects"].items()
    }

    driver.load_environment(
        {
            "task": {
                "instruction": "pick up the blue block and place it in the bowl",
                "target_color": "blue",
                "receptacle_name": "bowl",
            },
            "robot": {"ee_position": [0.0, -0.75], "gripper_closed": False, "held_object": None},
            "objects": {},
            "receptacles": {},
            "episode": {"step_count": 0, "last_reward": 0.0, "done": False, "last_info": {}},
        }
    )

    after_objects = {
        name: payload["position"].copy()
        for name, payload in driver.last_observation["objects"].items()
    }

    assert driver.env.task.target_color == "blue"
    assert after_objects.keys() == before_objects.keys()
    for name in before_objects:
        assert (after_objects[name] == before_objects[name]).all()
