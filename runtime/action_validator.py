from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_COLORS = {"red", "blue", "green"}
SUPPORTED_ACTIONS = {"reset", "env_step"}
WORKSPACE_LOW = np.array([-1.0, -1.0], dtype=float)
WORKSPACE_HIGH = np.array([1.0, 1.0], dtype=float)
MAX_STEP_DELTA = 0.06


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = "valid"


def validate_action(
    action_type: str,
    parameters: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> ValidationResult:
    if action_type not in SUPPORTED_ACTIONS:
        return ValidationResult(False, f"unsupported action_type {action_type!r}")
    if not isinstance(parameters, dict):
        return ValidationResult(False, "parameters must be an object")
    if action_type == "reset":
        return _validate_reset(parameters)
    if action_type == "env_step":
        return _validate_env_step(parameters, environment or {})
    return ValidationResult(False, f"no validator for action_type {action_type!r}")


def _validate_reset(parameters: dict[str, Any]) -> ValidationResult:
    target_color = str(parameters.get("target_color") or "").strip().lower()
    if target_color not in SUPPORTED_COLORS:
        return ValidationResult(False, f"target_color must be one of {sorted(SUPPORTED_COLORS)}")
    receptacle_name = str(parameters.get("receptacle_name") or "bowl")
    if receptacle_name != "bowl":
        return ValidationResult(False, "only receptacle_name='bowl' is supported")
    instruction = str(parameters.get("instruction") or "").strip()
    if not instruction:
        return ValidationResult(False, "instruction is required")
    return ValidationResult(True)


def _validate_env_step(parameters: dict[str, Any], environment: dict[str, Any]) -> ValidationResult:
    if "action" not in parameters:
        return ValidationResult(False, "parameters.action is required")
    try:
        action = np.asarray(parameters["action"], dtype=float)
    except (TypeError, ValueError):
        return ValidationResult(False, "parameters.action must be numeric")
    if action.shape != (3,):
        return ValidationResult(False, f"parameters.action must have shape (3,), got {tuple(action.shape)}")
    if not np.isfinite(action).all():
        return ValidationResult(False, "parameters.action must be finite")
    if np.any(np.abs(action[:2]) > MAX_STEP_DELTA + 1e-9):
        return ValidationResult(False, f"xy delta exceeds max step {MAX_STEP_DELTA}")

    robot = environment.get("robot", {}) if isinstance(environment, dict) else {}
    current = np.asarray(robot.get("ee_position", [0.0, -0.75]), dtype=float)
    proposed = current + action[:2]
    if np.any(proposed < WORKSPACE_LOW - 1e-9) or np.any(proposed > WORKSPACE_HIGH + 1e-9):
        return ValidationResult(False, "proposed end-effector position leaves workspace bounds")
    return ValidationResult(True)

