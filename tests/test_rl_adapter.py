import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.rl_policy import RLPolicy
from rl.gym_fake_manipulation import FakeManipulationGymEnv


def test_fake_manipulation_gym_env_step():
    env = FakeManipulationGymEnv(seed=0)
    observation, info = env.reset(options={"target_color": "blue"})
    next_observation, reward, terminated, truncated, step_info = env.step(np.array([0.0, 0.0, -1.0]))

    assert observation.shape == (env.observation_dim,)
    assert next_observation.shape == (env.observation_dim,)
    assert isinstance(reward, float)
    assert not (terminated and truncated)


def test_rl_policy_scripted_backend_returns_action():
    env = FakeManipulationGymEnv(seed=0)
    _, info = env.reset()
    action = RLPolicy(backend="scripted").act(info["observation"])

    assert action.shape == (3,)

