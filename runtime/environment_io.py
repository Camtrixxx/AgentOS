from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from envs.task_utils import parse_target_color


ENVIRONMENT_SCHEMA_VERSION = "embodied_lab.environment.v1"
FENCE_OPEN = "```json"
FENCE_CLOSE = "```"
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:
            pass

    return str(value)


def default_environment_doc() -> dict[str, Any]:
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "task": {
            "instruction": "pick up the red block and place it in the bowl",
            "target_color": "red",
            "receptacle_name": "bowl",
        },
        "robot": {
            "ee_position": [0.0, -0.75],
            "gripper_closed": False,
            "held_object": None,
        },
        "objects": {},
        "receptacles": {},
        "episode": {
            "step_count": 0,
            "success": False,
            "done": False,
            "last_reward": 0.0,
            "last_info": {},
        },
    }


def environment_doc_from_observation(
    observation: dict[str, Any],
    *,
    target_color: str | None = None,
    receptacle_name: str = "bowl",
    reward: float = 0.0,
    done: bool = False,
    info: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    info = info or {}
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "updated_at": updated_at or utc_now_iso(),
        "task": {
            "instruction": str(observation.get("instruction", "")),
            "target_color": target_color or infer_target_color(observation),
            "receptacle_name": receptacle_name,
        },
        "robot": {
            "ee_position": to_jsonable(observation.get("ee_position", [0.0, 0.0])),
            "gripper_closed": bool(observation.get("gripper_closed", False)),
            "held_object": observation.get("held_object"),
        },
        "objects": to_jsonable(observation.get("objects", {})),
        "receptacles": to_jsonable(observation.get("receptacles", {})),
        "episode": {
            "step_count": int(observation.get("step_count", 0)),
            "success": bool(info.get("success", False)),
            "done": bool(done),
            "last_reward": float(reward),
            "last_info": to_jsonable(info),
        },
    }


def infer_target_color(observation: dict[str, Any]) -> str:
    return parse_target_color(str(observation.get("instruction", "")))


def parse_environment_markdown(content: str) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return default_environment_doc()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return default_environment_doc()
    if not isinstance(payload, dict):
        return default_environment_doc()

    doc = default_environment_doc()
    doc.update(payload)
    return doc


def dump_environment_document(document: dict[str, Any] | None) -> str:
    payload = default_environment_doc()
    if isinstance(document, dict):
        payload.update(document)
    payload["schema_version"] = str(payload.get("schema_version") or ENVIRONMENT_SCHEMA_VERSION)
    payload["updated_at"] = str(payload.get("updated_at") or utc_now_iso())

    env_json = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False)
    return (
        "# Environment State\n\n"
        "Auto-updated by the embodied lab watchdog after action execution.\n\n"
        f"{FENCE_OPEN}\n{env_json}\n{FENCE_CLOSE}\n"
    )


def load_environment_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_environment_doc()
    return parse_environment_markdown(path.read_text(encoding="utf-8"))


def save_environment_document(path: Path, document: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_environment_document(document), encoding="utf-8")


def merge_runtime_state(document: dict[str, Any], runtime_state: dict[str, Any]) -> None:
    """Merge driver runtime state into an environment document in-place."""
    if not isinstance(document, dict) or not isinstance(runtime_state, dict):
        return
    document.setdefault("runtime", {})
    document["runtime"].update(to_jsonable(runtime_state))
    document["updated_at"] = utc_now_iso()

