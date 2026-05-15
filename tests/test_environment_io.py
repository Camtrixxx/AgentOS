import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.environment_io import (
    dump_environment_document,
    environment_doc_from_observation,
    observation_from_environment_doc,
    parse_environment_markdown,
)


def test_environment_doc_from_observation_is_jsonable():
    observation = {
        "instruction": "pick up the blue block and place it in the bowl",
        "ee_position": np.array([0.0, -0.75]),
        "gripper_closed": False,
        "held_object": None,
        "objects": {"blue_block": {"color": "blue", "position": np.array([0.4, 0.1])}},
        "receptacles": {"bowl": np.array([0.55, 0.65])},
        "step_count": 3,
    }

    document = environment_doc_from_observation(observation, reward=-0.01)

    assert document["task"]["target_color"] == "blue"
    assert document["robot"]["ee_position"] == [0.0, -0.75]
    assert document["objects"]["blue_block"]["position"] == [0.4, 0.1]
    assert document["episode"]["step_count"] == 3


def test_environment_markdown_round_trip():
    document = {"task": {"instruction": "test", "target_color": "green"}}
    markdown = dump_environment_document(document)
    parsed = parse_environment_markdown(markdown)

    assert parsed["schema_version"] == "embodied_lab.environment.v1"
    assert parsed["task"]["target_color"] == "green"


def test_observation_from_environment_doc_round_trip_shape():
    observation = {
        "instruction": "pick up the green block and place it in the bowl",
        "ee_position": np.array([0.1, -0.2]),
        "gripper_closed": True,
        "held_object": "green_block",
        "objects": {"green_block": {"color": "green", "position": np.array([0.1, -0.2])}},
        "receptacles": {"bowl": np.array([0.55, 0.65])},
        "step_count": 7,
    }
    document = environment_doc_from_observation(observation, reward=-0.01, info={"success": False})

    rebuilt = observation_from_environment_doc(document)

    assert rebuilt["instruction"] == observation["instruction"]
    assert rebuilt["ee_position"] == [0.1, -0.2]
    assert rebuilt["gripper_closed"] is True
    assert rebuilt["held_object"] == "green_block"
    assert rebuilt["objects"]["green_block"]["color"] == "green"
    assert rebuilt["step_count"] == 7
