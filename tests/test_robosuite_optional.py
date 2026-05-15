import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_robosuite_driver_imports_without_optional_dependency():
    from hal.robosuite_driver import RobosuiteDriver

    assert RobosuiteDriver.__name__ == "RobosuiteDriver"


def test_robosuite_env_reset_step_when_installed():
    pytest.importorskip("robosuite")

    import numpy as np

    from envs.robosuite_env import RobosuiteEnvAdapter, RobosuiteEnvConfig

    env = RobosuiteEnvAdapter(RobosuiteEnvConfig(task_name="Lift", robot="Panda"))
    try:
        obs = env.reset(instruction="lift the cube", target_color="red")
        assert "ee_position" in obs
        action = np.zeros(env.action_dim or 4, dtype=float)
        obs, reward, done, info = env.step(action)
        assert obs["step_count"] == 1
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_robosuite_lift_loop_tool_smoke_when_installed(tmp_path):
    pytest.importorskip("robosuite")

    from hal.drivers import load_driver
    from tools.embodied_tools import ResetTaskTool
    from tools.robosuite_tools import RobosuiteLiftLoopTool

    driver = load_driver("robosuite", task_name="Lift", robot="Panda", has_offscreen_renderer=False)
    reset = ResetTaskTool(tmp_path, driver=driver).run(
        {"instruction": "lift the cube", "target_color": "red", "receptacle_name": "bowl"}
    )
    response = RobosuiteLiftLoopTool(tmp_path, driver=driver).run({"max_steps": 3})

    assert reset.status.value == "success"
    assert response.status.value in {"success", "error"}
    assert response.data["step_records"]
    assert response.data["step_records"][-1]["step_count"] >= 1
