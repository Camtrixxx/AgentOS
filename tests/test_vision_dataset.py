import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.vision_demo_dataset import VisionDemoTransitionDataset


def test_vision_demo_dataset_loads_sample(tmp_path):
    episode_dir = tmp_path / "episode_000000"
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    np.save(images_dir / "000000.npy", image)
    transition = {
        "observation": {
            "instruction": "pick up the red block and place it in the bowl",
            "ee_position": [0.0, -0.75],
            "gripper_closed": False,
            "held_object": None,
            "objects": {
                "red_block": {"color": "red", "position": [-0.55, 0.15]},
                "blue_block": {"color": "blue", "position": [0.45, 0.1]},
                "green_block": {"color": "green", "position": [-0.1, 0.45]},
            },
            "receptacles": {"bowl": [0.55, 0.65]},
            "step_count": 0,
        },
        "image_path": "images/000000.npy",
        "action": [0.01, 0.02, -1.0],
    }
    (episode_dir / "transitions.jsonl").write_text(json.dumps(transition) + "\n", encoding="utf-8")

    dataset = VisionDemoTransitionDataset(tmp_path)
    image_tensor, task_tensor, state_tensor, action_tensor = dataset[0]

    assert image_tensor.shape == (3, 128, 128)
    assert task_tensor.tolist() == [1.0, 0.0, 0.0]
    assert state_tensor.shape == (15,)
    assert action_tensor.shape == (3,)
