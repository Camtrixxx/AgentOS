from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envs.task_utils import parse_target_color
from runtime.plan_utils import force_target_color
from runtime.planner import PlannedStep, Planner, RuleBasedPlanner, TaskPlan


SKILL_LIBRARY_SCHEMA_VERSION = "embodied_lab.skill_library.v1"
DEFAULT_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "default_skill.md"
JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class ParsedSkill:
    name: str
    description: str
    pattern: str
    parameters: dict[str, Any]
    target_color_param: str | None
    steps: list[dict[str, Any]]


class SkillLibraryPlanner:
    """Template-based planner backed by a markdown skill library."""

    def __init__(self, *, skill_path: str | Path | None = None, fallback: Planner | None = None):
        self.fallback = fallback or RuleBasedPlanner()
        self.skill_path = Path(skill_path) if skill_path is not None else DEFAULT_SKILL_PATH
        self._skills: dict[str, ParsedSkill] = {}
        if self.skill_path.exists():
            self._skills = parse_skills(self.skill_path.read_text(encoding="utf-8"))

    def plan(self, instruction: str, *, target_color: str | None = None) -> TaskPlan:
        for skill in self._skills.values():
            params = match_skill(skill, instruction)
            if params is None:
                continue
            if target_color is not None and skill.target_color_param:
                params[skill.target_color_param] = target_color
                if not validate_skill_params(skill, params):
                    continue
            return instantiate_skill(skill, params, instruction=instruction, target_color=target_color)
        return self.fallback.plan(instruction, target_color=target_color)


def parse_skills(markdown: str) -> dict[str, ParsedSkill]:
    for match in JSON_FENCE_RE.finditer(markdown):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if _is_skill_library_payload(payload):
            return _parse_skill_payload(payload)
    return {}


def _is_skill_library_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("skills"), list)
        and (
            payload.get("schema_version") == SKILL_LIBRARY_SCHEMA_VERSION
            or any(isinstance(item, dict) and "pattern" in item and "steps" in item for item in payload["skills"])
        )
    )


def _parse_skill_payload(payload: dict[str, Any]) -> dict[str, ParsedSkill]:
    skills_payload = payload.get("skills", [])
    skills: dict[str, ParsedSkill] = {}
    for item in skills_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        pattern = str(item.get("pattern") or "").strip()
        steps = item.get("steps", [])
        if not name or not pattern or not isinstance(steps, list):
            continue
        skills[name] = ParsedSkill(
            name=name,
            description=str(item.get("description") or ""),
            pattern=pattern,
            parameters=item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
            target_color_param=str(item.get("target_color_param") or "") or None,
            steps=[step for step in steps if isinstance(step, dict)],
        )
    return skills


def match_skill(skill: ParsedSkill, instruction: str) -> dict[str, str] | None:
    pattern = _compile_pattern(skill.pattern)
    match = pattern.fullmatch(_normalize_text(instruction))
    if match is None:
        return None
    params = {key: value.lower() for key, value in match.groupdict().items()}
    return params if validate_skill_params(skill, params) else None


def instantiate_skill(
    skill: ParsedSkill,
    params: dict[str, str],
    *,
    instruction: str,
    target_color: str | None = None,
) -> TaskPlan:
    target_param = skill.target_color_param
    color = target_color or (params.get(target_param) if target_param else None) or parse_target_color(instruction)
    normalized_instruction = _replace_placeholders(skill.pattern, params)
    steps = [
        PlannedStep(
            tool=str(step.get("tool") or ""),
            parameters=_replace_placeholders(step.get("parameters", {}), params),
            description=_replace_placeholders(str(step.get("description") or ""), params),
        )
        for step in skill.steps
    ]
    return force_target_color(TaskPlan(normalized_instruction, color, steps), color)


def validate_skill_params(skill: ParsedSkill, params: dict[str, str]) -> bool:
    for name, spec in skill.parameters.items():
        if name not in params:
            return False
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "string" and not isinstance(params[name], str):
            return False
        values = spec.get("values")
        if isinstance(values, list) and params[name] not in {str(value).lower() for value in values}:
            return False
    return True


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    normalized = _normalize_text(pattern)
    parts: list[str] = []
    last = 0
    for match in PLACEHOLDER_RE.finditer(normalized):
        parts.append(re.escape(normalized[last : match.start()]))
        name = match.group(1)
        parts.append(f"(?P<{name}>[a-zA-Z0-9_-]+)")
        last = match.end()
    parts.append(re.escape(normalized[last:]))
    return re.compile("".join(parts), flags=re.IGNORECASE)


def _normalize_text(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    return collapsed.rstrip(".!?")


def _replace_placeholders(value: Any, params: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, item in params.items():
            result = result.replace("{" + key + "}", str(item))
        return result
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, params) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item, params) for item in value]
    return value

