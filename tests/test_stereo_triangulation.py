import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_fake_pipeline import make_synthetic_hand, project
from perception.stereo_triangulation import StereoCameraConfig, StereoHandTriangulation


def test_stereo_triangulation_reconstructs_synthetic_points():
    k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    config = StereoCameraConfig(
        k_left=k,
        k_right=k.copy(),
        r_left_to_right=np.eye(3),
        t_left_to_right=np.array([-0.08, 0.0, 0.0]),
    )

    hand_3d = make_synthetic_hand()
    uv_left = project(hand_3d, k, np.eye(3), np.zeros(3))
    uv_right = project(hand_3d, k, config.r_left_to_right, config.t_left_to_right)
    reconstructed = StereoHandTriangulation(config).triangulate(uv_left, uv_right)

    np.testing.assert_allclose(reconstructed, hand_3d, atol=1e-8)

