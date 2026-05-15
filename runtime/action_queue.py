from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from runtime.environment_io import to_jsonable
from runtime.file_io import atomic_write_text


ACTION_QUEUE_SCHEMA_VERSION = "embodied_lab.action_queue.v1"
FENCE_OPEN = "```json"
FENCE_CLOSE = "```"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
KNOWN_STATUSES = {"pending", "running", *TERMINAL_STATUSES}


def parse_action_markdown(content: str) -> dict[str, Any] | None:
    """Parse the first JSON fenced block from ACTION.md content."""

    text = content.strip()
    if not text:
        return None
    try:
        _, json_block = text.split(FENCE_OPEN, 1)
        json_block, _ = json_block.split(FENCE_CLOSE, 1)
        payload = json.loads(json_block)
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def normalize_action_item(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    action_type = str(payload.get("action_type") or "").strip()
    if not action_type:
        return None

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        return None

    status = str(payload.get("status") or "pending").strip().lower() or "pending"
    if status not in KNOWN_STATUSES:
        return None
    item: dict[str, Any] = {
        "id": str(payload.get("id") or uuid.uuid4().hex[:12]),
        "action_type": action_type,
        "parameters": parameters,
        "status": status,
        "created_at": str(payload.get("created_at") or ""),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "attempt_count": _safe_int(payload.get("attempt_count"), default=0),
    }
    if "result" in payload:
        item["result"] = payload["result"]
    if "error" in payload:
        item["error"] = payload["error"]
    return item


def normalize_action_document(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return empty_action_document()

    actions = payload.get("actions")
    if isinstance(actions, list):
        normalized_actions = []
        for action in actions:
            normalized = normalize_action_item(action)
            if normalized is None:
                return empty_action_document()
            normalized_actions.append(normalized)
        return {
            "schema_version": str(payload.get("schema_version") or ACTION_QUEUE_SCHEMA_VERSION),
            "actions": normalized_actions,
        }

    normalized = normalize_action_item(payload)
    if normalized is None:
        return empty_action_document()
    return {"schema_version": ACTION_QUEUE_SCHEMA_VERSION, "actions": [normalized]}


def empty_action_document() -> dict[str, Any]:
    return {"schema_version": ACTION_QUEUE_SCHEMA_VERSION, "actions": []}


def first_pending_action(document: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for index, item in enumerate(document.get("actions", [])):
        if str(item.get("status") or "pending").lower() == "pending":
            return index, item
    return None


def append_action(
    document: dict[str, Any] | None,
    *,
    action_type: str,
    parameters: dict[str, Any] | None = None,
    action_id: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_action_document(document)
    from runtime.environment_io import utc_now_iso

    normalized["actions"].append(
        {
            "id": action_id or uuid.uuid4().hex[:12],
            "action_type": action_type,
            "parameters": parameters or {},
            "status": "pending",
            "created_at": utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "attempt_count": 0,
            "result": None,
            "error": None,
        }
    )
    return normalized


def mark_action_running(document: dict[str, Any], action_index: int) -> dict[str, Any]:
    from runtime.environment_io import utc_now_iso

    action = document["actions"][action_index]
    action["status"] = "running"
    action["started_at"] = utc_now_iso()
    action["finished_at"] = None
    action["attempt_count"] = _safe_int(action.get("attempt_count"), default=0) + 1
    action["error"] = None
    return document


def mark_action_finished(document: dict[str, Any], action_index: int, result: Any) -> dict[str, Any]:
    from runtime.environment_io import utc_now_iso

    status = infer_terminal_status(result)
    action = document["actions"][action_index]
    action["status"] = status
    action["finished_at"] = utc_now_iso()
    action["result"] = result
    action["error"] = _result_error(result) if status == "failed" else None
    return document


def dump_action_document(document: dict[str, Any] | None) -> str:
    normalized = normalize_action_document(document)
    payload = json.dumps(to_jsonable(normalized), indent=2, ensure_ascii=False)
    return (
        "# Action Queue\n\n"
        "Watchdog executes the first action with `status = pending`.\n\n"
        f"{FENCE_OPEN}\n{payload}\n{FENCE_CLOSE}\n"
    )


def load_action_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_action_document()
    return normalize_action_document(parse_action_markdown(path.read_text(encoding="utf-8")))


def save_action_document(path: Path, document: dict[str, Any] | None) -> None:
    atomic_write_text(path, dump_action_document(document), encoding="utf-8")


def infer_terminal_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("success") is False or result.get("error"):
            return "failed"
        if result.get("cancelled"):
            return "cancelled"
        return "completed"

    lowered = str(result).strip().lower()
    if lowered.startswith("error:") or " failed" in lowered or lowered.startswith("unknown action"):
        return "failed"
    if "cancelled" in lowered or "canceled" in lowered:
        return "cancelled"
    return "completed"


def _result_error(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return {
            "code": str(result.get("code") or result.get("error") or "action_failed"),
            "message": str(result.get("message") or result.get("error") or "action failed"),
        }
    return {"code": "action_failed", "message": str(result)}


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
