import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.action_validator import validate_action


def test_validator_accepts_small_env_step():
    environment = {"robot": {"ee_position": [0.0, -0.75]}}

    result = validate_action("env_step", {"action": [0.02, 0.0, -1.0]}, environment)

    assert result.valid


def test_validator_rejects_oversized_delta():
    environment = {"robot": {"ee_position": [0.0, -0.75]}}

    result = validate_action("env_step", {"action": [0.2, 0.0, -1.0]}, environment)

    assert not result.valid
    assert "max step" in result.reason


def test_validator_rejects_unknown_reset_color():
    result = validate_action(
        "reset",
        {"instruction": "pick up the yellow block", "target_color": "yellow", "receptacle_name": "bowl"},
    )

    assert not result.valid
    assert "target_color" in result.reason

