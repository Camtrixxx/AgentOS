from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hal.base_driver import BaseDriver
from runtime.action_queue import (
    first_pending_action,
    infer_terminal_status,
)
from runtime.action_validator import validate_action
from runtime.environment_io import merge_runtime_state
from runtime.repository import WorkspaceRepository


def poll_once(driver: BaseDriver, repo: WorkspaceRepository) -> dict[str, Any] | None:
    """Execute the first pending action and update workspace files."""

    driver.health_check()
    env_doc = driver.get_environment()
    merge_runtime_state(env_doc, driver.get_runtime_state())
    repo.save_environment(env_doc)

    document = repo.get_actions()
    pending = first_pending_action(document)
    if pending is None:
        return None

    action_index, action = pending
    action_type = str(action.get("action_type") or "unknown")
    parameters = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
    environment = repo.get_environment()
    validation = validate_action(action_type, parameters, environment)
    if validation.valid:
        result = driver.execute_action(action_type, parameters)
    else:
        result = {
            "success": False,
            "message": f"critic rejected action: {validation.reason}",
            "action_type": action_type,
        }
    document["actions"][action_index]["status"] = infer_terminal_status(result)
    document["actions"][action_index]["result"] = result
    repo.save_actions(document)

    env_doc = driver.get_environment()
    merge_runtime_state(env_doc, driver.get_runtime_state())
    repo.save_environment(env_doc)
    return result


def run_watchdog(
    driver: BaseDriver,
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

    with driver:
        env_doc = driver.get_environment()
        merge_runtime_state(env_doc, driver.get_runtime_state())
        repo.save_environment(env_doc)
        while True:
            poll_once(driver, repo)
            if once:
                return
            time.sleep(poll_interval)
