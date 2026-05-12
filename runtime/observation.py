from __future__ import annotations


def observation_from_environment(environment: dict) -> dict:
    task = environment.get("task", {})
    robot = environment.get("robot", {})
    return {
        "instruction": task.get("instruction", ""),
        "ee_position": robot.get("ee_position", [0.0, -0.75]),
        "gripper_closed": bool(robot.get("gripper_closed", False)),
        "held_object": robot.get("held_object"),
        "objects": {
            name: {"color": obj.get("color"), "position": obj.get("position", [0.0, 0.0])}
            for name, obj in environment.get("objects", {}).items()
        },
        "receptacles": {
            name: payload.get("position", payload) if isinstance(payload, dict) else payload
            for name, payload in environment.get("receptacles", {}).items()
        },
        "step_count": int(environment.get("episode", {}).get("step_count", 0)),
    }

