import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hal.vla_adapter import VLAObservation
from agent.vla_smolvla_backend import SmolVLABackend


def test_smolvla_backend_dry_run_predicts_action():
    backend = SmolVLABackend(dry_run=True)
    observation = VLAObservation(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        instruction="pick up the red block and place it in the bowl",
        state={
            "ee_position": np.array([0.0, -0.75]),
            "gripper_closed": False,
            "held_object": None,
            "objects": {"red_block": {"color": "red", "position": [-0.55, 0.15]}},
            "receptacles": {"bowl": [0.55, 0.65]},
            "step_count": 0,
        },
    )

    action = backend.predict(observation)

    assert action.ee_delta.shape == (2,)
    assert action.raw["backend"] == "smolvla"
    assert action.raw["mode"] == "mock_fallback"

