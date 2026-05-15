from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from runtime.planner import TaskPlan
from runtime.skill_planner import JSON_FENCE_RE, SKILL_LIBRARY_SCHEMA_VERSION, parse_skills


SUPPORTED_COLORS = ("blue", "green", "red")


def record_plan_as_skill(
    plan: TaskPlan,
    instruction: str,
    skill_path: str | Path,
    *,
    skill_name: str | None = None,
    color_param: str = "color",
) -> bool:
    """Record a successful TaskPlan into a markdown skill library.

    Returns True when a new skill is appended. Existing skills with the same
    derived pattern are left untouched.
    """

    path = Path(skill_path)
    markdown = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = derive_pattern(instruction, color_param=color_param)
    existing = parse_skills(markdown)
    if any(_normalize_pattern(skill.pattern) == _normalize_pattern(pattern) for skill in existing.values()):
        return False

    payload = _load_skill_payload(markdown)
    skills = payload.setdefault("skills", [])
    if not isinstance(skills, list):
        skills = []
        payload["skills"] = skills

    name = _unique_skill_name(skill_name or _derive_skill_name(pattern), skills)
    skills.append(_plan_to_skill_dict(plan, instruction, name=name, pattern=pattern, color_param=color_param))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_skill_library(payload), encoding="utf-8")
    return True


def derive_pattern(instruction: str, *, color_param: str = "color") -> str:
    pattern = instruction.strip()
    for color in SUPPORTED_COLORS:
        pattern = re.sub(rf"\b{re.escape(color)}\b", "{" + color_param + "}", pattern, flags=re.IGNORECASE)
    return pattern


def _plan_to_skill_dict(
    plan: TaskPlan,
    instruction: str,
    *,
    name: str,
    pattern: str,
    color_param: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Auto-recorded successful workflow: {instruction.strip()}",
        "pattern": pattern,
        "parameters": {
            color_param: {
                "type": "string",
                "values": sorted(SUPPORTED_COLORS),
            }
        },
        "target_color_param": color_param,
        "steps": [
            {
                "tool": step.tool,
                "parameters": _template_colors(step.parameters, color_param=color_param),
                "description": _template_colors(step.description, color_param=color_param),
            }
            for step in plan.steps
        ],
    }


def _load_skill_payload(markdown: str) -> dict[str, Any]:
    for match in JSON_FENCE_RE.finditer(markdown):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
            payload.setdefault("schema_version", SKILL_LIBRARY_SCHEMA_VERSION)
            return payload
    return {"schema_version": SKILL_LIBRARY_SCHEMA_VERSION, "skills": []}


def _template_colors(value: Any, *, color_param: str) -> Any:
    if isinstance(value, str):
        result = value
        for color in SUPPORTED_COLORS:
            result = re.sub(rf"\b{re.escape(color)}\b", "{" + color_param + "}", result, flags=re.IGNORECASE)
        return result
    if isinstance(value, dict):
        return {key: _template_colors(item, color_param=color_param) for key, item in value.items()}
    if isinstance(value, list):
        return [_template_colors(item, color_param=color_param) for item in value]
    return value


def _format_skill_library(payload: dict[str, Any]) -> str:
    return "# Skill Library\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```\n"


def _derive_skill_name(pattern: str) -> str:
    lowered = pattern.lower()
    if "pick up" in lowered and "place" in lowered and "bowl" in lowered:
        return "pick_place"
    name = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", r"\1", lowered)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name[:64] or "recorded_skill"


def _unique_skill_name(name: str, skills: list[Any]) -> str:
    existing = {str(item.get("name")) for item in skills if isinstance(item, dict)}
    if name not in existing:
        return name
    suffix = 2
    while f"{name}_{suffix}" in existing:
        suffix += 1
    return f"{name}_{suffix}"


def _normalize_pattern(pattern: str) -> str:
    return " ".join(pattern.strip().lower().split()).rstrip(".!?")
