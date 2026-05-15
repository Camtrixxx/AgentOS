import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.vla_adapter import FakeEnvVLAAdapter, VLAAction
from agent.agent_loop import run_episode
from agent.vla_policy import VLAPolicy
from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec
from agent.vla_smolvla_backend import SmolVLABackend


def test_vla_adapter_converts_action():
    adapter = FakeEnvVLAAdapter()

    action = adapter.action_from_vla(VLAAction(ee_delta=np.array([0.1, -0.2]), gripper=1.0))

    np.testing.assert_allclose(action, np.array([0.1, -0.2, 1.0]))


def test_mock_vla_policy_solves_randomized_layout():
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        include_image=True,
        randomize_layout=True,
    )
    env = FakeManipulationEnv(config=config, seed=4)
    policy = VLAPolicy()
    task = TaskSpec("pick up the blue block and place it in the bowl", "blue")

    result = run_episode(env, policy, task=task, max_steps=100)

    assert result.success


def test_vla_policy_accepts_smolvla_dry_run_backend():
    env = FakeManipulationEnv(seed=0)
    observation = env.reset(TaskSpec("pick up the red block and place it in the bowl", "red"))
    policy = VLAPolicy(backend=SmolVLABackend(dry_run=True))

    action = policy.act(observation)

    assert action.shape == (3,)
