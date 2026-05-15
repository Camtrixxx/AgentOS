import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.scripted_policy import ScriptedPickPlacePolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv
from runtime.executor import execute_task_plan
from runtime.planner import RuleBasedPlanner


def test_execute_task_plan_writes_plan_and_report(tmp_path):
    workspace = tmp_path / "workspace"
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
    )
    env = FakeManipulationEnv(config=config, seed=0)
    policy = ScriptedPickPlacePolicy()
    plan = RuleBasedPlanner().plan("pick up the red block and place it in the bowl")

    result = execute_task_plan(
        plan,
        env=env,
        policy=policy,
        workspace=workspace,
        max_steps=80,
        render_output=tmp_path / "render.ppm",
    )

    assert result.success
    assert result.steps < 80
    assert (workspace / "PLAN.md").exists()
    assert (workspace / "REPORT.md").exists()
    assert (tmp_path / "render.ppm").exists()
