import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recorders.inspector import inspect_demo_dataset, write_inspection_report


def test_inspect_demo_dataset_reports_stats(tmp_path):
    episode = tmp_path / "episode_000000"
    images = episode / "images"
    images.mkdir(parents=True)
    np.save(images / "000000.npy", np.zeros((128, 128, 3), dtype=np.uint8))
    np.save(images / "000000_next.npy", np.zeros((128, 128, 3), dtype=np.uint8))
    metadata = {"summary": {"success": True}}
    transition = {
        "observation": {"instruction": "pick up the blue block"},
        "image_path": "images/000000.npy",
        "action": [0.01, 0.02, -1.0],
        "reward": -0.01,
        "next_observation": {"instruction": "pick up the blue block"},
        "next_image_path": "images/000000_next.npy",
        "done": False,
        "info": {},
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (episode / "transitions.jsonl").write_text(json.dumps(transition) + "\n", encoding="utf-8")

    inspection = inspect_demo_dataset(tmp_path, expect_images=True)
    json_path, md_path = write_inspection_report(inspection, tmp_path / "reports")

    assert inspection.ok
    assert inspection.success_rate == 1.0
    assert inspection.target_color_counts["blue"] == 1
    assert json_path.exists()
    assert md_path.exists()

