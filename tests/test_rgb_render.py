import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv, TaskSpec


def test_render_rgb_shape_and_dtype():
    env = FakeManipulationEnv(seed=0)
    env.reset(TaskSpec("pick up the red block and place it in the bowl", "red"))

    image = env.render_rgb()

    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8


def test_image_observation_can_be_enabled():
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        include_image=True,
    )
    env = FakeManipulationEnv(config=config, seed=0)

    observation = env.reset(TaskSpec("pick up the blue block and place it in the bowl", "blue"))

    assert "image" in observation
    assert observation["image"].shape == (128, 128, 3)


def test_randomized_layout_changes_positions():
    config = FakeManipulationConfig(
        workspace_low=np.array([-1.0, -1.0], dtype=float),
        workspace_high=np.array([1.0, 1.0], dtype=float),
        randomize_layout=True,
    )
    env_a = FakeManipulationEnv(config=config, seed=1)
    env_b = FakeManipulationEnv(config=config, seed=2)

    obs_a = env_a.reset(TaskSpec("pick up the red block and place it in the bowl", "red"))
    obs_b = env_b.reset(TaskSpec("pick up the red block and place it in the bowl", "red"))

    assert obs_a["objects"]["red_block"]["position"].tolist() != obs_b["objects"]["red_block"]["position"].tolist()
