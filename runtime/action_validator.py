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
    capabilities = _capabilities_from_environment(environment or {})
    supported_actions = set(capabilities.get("supported_actions") or SUPPORTED_ACTIONS)
    if action_type not in supported_actions:
        return ValidationResult(False, f"unsupported action_type {action_type!r}")
    if not isinstance(parameters, dict):
        return ValidationResult(False, "parameters must be an object")
    if action_type == "reset":
        return _validate_reset(parameters, capabilities)
    if action_type == "env_step":
        return _validate_env_step(parameters, environment or {}, capabilities)
    return ValidationResult(False, f"no validator for action_type {action_type!r}")


def _validate_reset(parameters: dict[str, Any], capabilities: dict[str, Any]) -> ValidationResult:
    target_color = str(parameters.get("target_color") or "").strip().lower()
    supported_colors = set(capabilities.get("supported_colors") or SUPPORTED_COLORS)
    if target_color not in supported_colors:
        return ValidationResult(False, f"target_color must be one of {sorted(supported_colors)}")
    receptacle_name = str(parameters.get("receptacle_name") or "bowl")
    supported_receptacles = set(capabilities.get("receptacles") or ["bowl"])
    if receptacle_name not in supported_receptacles:
        return ValidationResult(False, f"receptacle_name must be one of {sorted(supported_receptacles)}")
    instruction = str(parameters.get("instruction") or "").strip()
    if not instruction:
        return ValidationResult(False, "instruction is required")
    return ValidationResult(True)


def _validate_env_step(
    parameters: dict[str, Any],
    environment: dict[str, Any],
    capabilities: dict[str, Any],
) -> ValidationResult:
    if "action" not in parameters:
        return ValidationResult(False, "parameters.action is required")
    try:
        action = np.asarray(parameters["action"], dtype=float)
    except (TypeError, ValueError):
        return ValidationResult(False, "parameters.action must be numeric")
    action_dims = _supported_action_dims(capabilities)
    if action.shape not in {(dim,) for dim in action_dims}:
        return ValidationResult(False, f"parameters.action must have one of shapes {sorted(action_dims)}, got {tuple(action.shape)}")
    if not np.isfinite(action).all():
        return ValidationResult(False, "parameters.action must be finite")
    max_step_delta = float(capabilities.get("max_step_delta") or MAX_STEP_DELTA)
    delta_dim = _delta_dim(action, environment, capabilities)
    if delta_dim and np.any(np.abs(action[:delta_dim]) > max_step_delta + 1e-9):
        return ValidationResult(False, f"xy delta exceeds max step {max_step_delta}")

    robot = environment.get("robot", {}) if isinstance(environment, dict) else {}
    current = np.asarray(robot.get("ee_position", [0.0, -0.75]), dtype=float)
    workspace_low = np.asarray(capabilities.get("workspace_low") or WORKSPACE_LOW, dtype=float)
    workspace_high = np.asarray(capabilities.get("workspace_high") or WORKSPACE_HIGH, dtype=float)
    bounds_dim = min(delta_dim, current.size, workspace_low.size, workspace_high.size)
    proposed = current[:bounds_dim] + action[:bounds_dim]
    if np.any(proposed < workspace_low[:bounds_dim] - 1e-9) or np.any(proposed > workspace_high[:bounds_dim] + 1e-9):
        return ValidationResult(False, "proposed end-effector position leaves workspace bounds")
    return ValidationResult(True)


def _capabilities_from_environment(environment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(environment, dict):
        return {}
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict):
        return {}
    capabilities = runtime.get("capabilities")
    return capabilities if isinstance(capabilities, dict) else {}


def _supported_action_dims(capabilities: dict[str, Any]) -> list[int]:
    raw_dims = capabilities.get("action_dims")
    if isinstance(raw_dims, list):
        dims = [int(dim) for dim in raw_dims if isinstance(dim, (int, float, str)) and str(dim).isdigit()]
        if dims:
            return dims
    action_dim = capabilities.get("action_dim")
    if isinstance(action_dim, (int, float, str)) and str(action_dim).isdigit():
        return [int(action_dim)]
    return [3]


def _delta_dim(action: np.ndarray, environment: dict[str, Any], capabilities: dict[str, Any]) -> int:
    robot = environment.get("robot", {}) if isinstance(environment, dict) else {}
    current = np.asarray(robot.get("ee_position", [0.0, -0.75]), dtype=float).reshape(-1)
    workspace_low = np.asarray(capabilities.get("workspace_low") or WORKSPACE_LOW, dtype=float).reshape(-1)
    workspace_high = np.asarray(capabilities.get("workspace_high") or WORKSPACE_HIGH, dtype=float).reshape(-1)
    return min(max(action.size - 1, 0), current.size, workspace_low.size, workspace_high.size)
