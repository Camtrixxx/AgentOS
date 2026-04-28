import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.fake_manipulation_env import FakeManipulationEnv, TaskSpec
from learning.features import FEATURE_DIM, extract_state_features


def test_extract_state_features_shape():
    env = FakeManipulationEnv(seed=0)
    observation = env.reset(TaskSpec("pick up the red block and place it in the bowl", "red"))

    features = extract_state_features(observation)

    assert features.shape == (FEATURE_DIM,)

