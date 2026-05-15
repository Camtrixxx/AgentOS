from __future__ import annotations

from typing import Any


COLORS = ("red", "blue", "green")


def parse_target_color(instruction: str) -> str:
    lowered = instruction.lower()
    for color in COLORS:
        if color in lowered:
            return color
    return "red"


def find_object_by_color(objects: dict[str, Any], color: str) -> str:
    for name, obj in objects.items():
        if obj.get("color") == color:
            return name
    raise ValueError(f"No object with color {color!r}")
