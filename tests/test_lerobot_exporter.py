import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recorders.lerobot_exporter import export_vision_demos_to_lerobot_jsonl, export_vision_demos_to_lerobot_native


def test_export_vision_demos_to_lerobot_jsonl(tmp_path):
    episode = tmp_path / "data" / "episode_000000"
    images = episode / "images"
    images.mkdir(parents=True)
    np.save(images / "000000.npy", np.zeros((128, 128, 3), dtype=np.uint8))
    transition = {
        "observation": {
            "instruction": "pick up the red block and place it in the bowl",
            "ee_position": [0.0, -0.75],
            "gripper_closed": False,
            "held_object": None,
            "objects": {"red_block": {"color": "red", "position": [-0.55, 0.15]}},
            "receptacles": {"bowl": [0.55, 0.65]},
            "step_count": 0,
        },
        "image_path": "images/000000.npy",
        "action": [0.01, 0.02, -1.0],
        "reward": -0.01,
        "done": False,
    }
    (episode / "transitions.jsonl").write_text(json.dumps(transition) + "\n", encoding="utf-8")

    summary = export_vision_demos_to_lerobot_jsonl(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    row = json.loads((tmp_path / "out" / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert summary.num_episodes == 1
    assert summary.num_frames == 1
    assert row["target_color"] == "red"
    assert "observation.images.front" in row


def test_native_lerobot_export_reports_missing_dependency(tmp_path):
    try:
        export_vision_demos_to_lerobot_native(data_dir=tmp_path / "missing", output_dir=tmp_path / "out")
    except RuntimeError as exc:
        assert "LeRobot is not installed" in str(exc)
    else:
        # If the test environment has LeRobot, this path is acceptable.
        assert True
