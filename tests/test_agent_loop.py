import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_loop import run_episode
from agent.scripted_policy import ScriptedPickPlacePolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec
import numpy as np


def test_scripted_agent_solves_fake_pick_place():
    env = FakeManipulationEnv(seed=0)
    policy = ScriptedPickPlacePolicy()
    task = TaskSpec("pick up the red block and place it in the bowl", "red")

    result = run_episode(env, policy, task=task, max_steps=80)

    assert result.success
    assert result.steps < 80


def test_scripted_agent_solves_randomized_layout():
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        randomize_layout=True,
    )
    env = FakeManipulationEnv(config=config, seed=3)
    policy = ScriptedPickPlacePolicy()
    task = TaskSpec("pick up the green block and place it in the bowl", "green")

    result = run_episode(env, policy, task=task, max_steps=100)

    assert result.success
