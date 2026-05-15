from __future__ import annotations

from pathlib import Path
from typing import Any

from hal.base_driver import DriverStateError, RuntimeDriver
from runtime.action_queue import (
    first_pending_action,
    mark_action_finished,
    mark_action_running,
)
from runtime.action_validator import validate_action
from runtime.environment_io import merge_runtime_state
from runtime.file_watcher import FileWatcher
from runtime.repository import WorkspaceRepository


def poll_once(driver: RuntimeDriver, repo: WorkspaceRepository) -> dict[str, Any] | None:
    """Execute the first pending action and update workspace files."""

    try:
        healthy = driver.health_check()
    except DriverStateError as exc:
        raise RuntimeError(f"driver health_check failed: {exc}") from exc
    if not healthy:
        raise RuntimeError("driver health_check failed")
    env_doc = driver.get_environment()
    merge_runtime_state(env_doc, driver.get_runtime_state())
    repo.save_environment(env_doc)

    claimed = _claim_first_pending_action(repo)
    if claimed is None:
        return None

    action_id, action = claimed
    action_type = str(action.get("action_type") or "unknown")
    parameters = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
    environment = repo.get_environment()
    validation = validate_action(action_type, parameters, environment)
    if validation.valid:
        try:
            result = driver.execute_action(action_type, parameters)
        except DriverStateError as exc:
            result = {
                "success": False,
                "code": "driver_not_ready",
                "message": str(exc),
                "action_type": action_type,
            }
        except Exception as exc:
            result = {
                "success": False,
                "code": "driver_exception",
                "message": f"driver raised {type(exc).__name__}: {exc}",
                "action_type": action_type,
            }
    else:
        result = {
            "success": False,
            "code": "critic_rejected_action",
            "message": f"critic rejected action: {validation.reason}",
            "action_type": action_type,
        }
    _finish_action(repo, action_id, result)

    env_doc = driver.get_environment()
    merge_runtime_state(env_doc, driver.get_runtime_state())
    repo.save_environment(env_doc)
    return result


def _claim_first_pending_action(repo: WorkspaceRepository) -> tuple[str, dict[str, Any]] | None:
    def claim(document: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        pending = first_pending_action(document)
        if pending is None:
            return None
        action_index, action = pending
        mark_action_running(document, action_index)
        return str(action["id"]), dict(action)

    return repo.update_actions(claim)


def _finish_action(repo: WorkspaceRepository, action_id: str, result: dict[str, Any]) -> None:
    def finish(document: dict[str, Any]) -> None:
        for index, action in enumerate(document.get("actions", [])):
            if str(action.get("id")) == action_id:
                mark_action_finished(document, index, result)
                return
        document.setdefault("actions", []).append(
            {
                "id": action_id,
                "action_type": "unknown",
                "parameters": {},
                "status": "failed",
                "result": result,
                "error": {"code": "action_missing", "message": "claimed action disappeared before finish"},
            }
        )

    repo.update_actions(finish)


def run_watchdog(
    driver: RuntimeDriver,
    *,
    workspace: str | Path = "workspace",
    poll_interval: float = 1.0,
    once: bool = False,
    initialize: bool = True,
) -> None:
    repo = WorkspaceRepository(workspace)
    if initialize:
        repo.initialize()
    driver.load_environment(repo.get_environment())

    driver.connect()
    try:
        env_doc = driver.get_environment()
        merge_runtime_state(env_doc, driver.get_runtime_state())
        repo.save_environment(env_doc)
        watcher = FileWatcher()
        try:
            while True:
                poll_once(driver, repo)
                if once:
                    return
                watcher.wait_for_change(repo.paths.action, timeout=poll_interval)
        finally:
            watcher.close()
    finally:
        driver.close()
