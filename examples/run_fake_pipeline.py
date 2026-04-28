from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.fake_robot_backend import FakeRobotBackend
from control.safety_limiter import SafetyConfig, SafetyLimiter
from perception.stereo_triangulation import StereoCameraConfig, StereoHandTriangulation
from retargeting.simple_hand_retargeter import SimpleHandRetargeter


def make_synthetic_hand() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=float)
    joints[:, 0] = np.linspace(-0.06, 0.06, 21)
    joints[:, 1] = 0.04 * np.sin(np.linspace(0, np.pi, 21))
    joints[:, 2] = 0.65 + 0.02 * np.cos(np.linspace(0, np.pi, 21))
    return joints


def project(points_3d: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    points_cam = (r @ points_3d.T + t.reshape(3, 1)).T
    uvw = (k @ points_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def main() -> None:
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

    triangulator = StereoHandTriangulation(config)
    retargeter = SimpleHandRetargeter()
    robot = FakeRobotBackend(["thumb", "index", "middle", "ring", "little", "spread"])
    limiter = SafetyLimiter(
        SafetyConfig(
            joint_lower=np.zeros(6),
            joint_upper=np.ones(6),
            max_delta_per_step=0.08,
        )
    )

    reconstructed = triangulator.triangulate(uv_left, uv_right)
    command = retargeter.retarget(reconstructed)
    safe_command = limiter.limit(command)
    state = robot.send_joint_command(safe_command)

    print("reconstruction_error_m", float(np.abs(reconstructed - hand_3d).mean()))
    print("safe_command", np.round(safe_command, 4).tolist())
    print("robot_frame_id", state.frame_id)


if __name__ == "__main__":
    main()

