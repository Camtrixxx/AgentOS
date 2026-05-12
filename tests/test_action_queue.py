import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.action_queue import append_action, dump_action_document, first_pending_action, parse_action_markdown


def test_action_queue_round_trip_markdown():
    document = append_action(None, action_type="env_step", parameters={"action": [0.01, 0.0, -1.0]})
    markdown = dump_action_document(document)
    parsed = parse_action_markdown(markdown)

    assert parsed is not None
    assert parsed["schema_version"] == "embodied_lab.action_queue.v1"
    assert parsed["actions"][0]["action_type"] == "env_step"
    assert parsed["actions"][0]["status"] == "pending"


def test_first_pending_action_skips_completed_actions():
    document = {
        "actions": [
            {"id": "done", "action_type": "reset", "parameters": {}, "status": "completed"},
            {"id": "next", "action_type": "env_step", "parameters": {"action": [0, 0, 1]}, "status": "pending"},
        ]
    }

    pending = first_pending_action(document)

    assert pending is not None
    index, action = pending
    assert index == 1
    assert action["id"] == "next"

