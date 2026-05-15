"""
hal/base_driver.py

Abstract base class for all robot body drivers.

Every hardware or simulation embodiment MUST subclass `BaseDriver` and
implement the four abstract methods. The Watchdog loads a driver by
name and interacts with it exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class DriverState(str, Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    EXECUTING = "executing"
    FAULT = "fault"
    CLOSED = "closed"


class DriverStateError(RuntimeError):
    """Raised when a driver action is attempted in an invalid state."""


class CommandDriver(Protocol):
    def execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        ...


class QueryDriver(Protocol):
    def load_environment(self, environment: dict[str, Any]) -> None:
        ...

    def health_check(self) -> bool:
        ...

    def get_environment(self) -> dict[str, Any]:
        ...

    def get_runtime_state(self) -> dict[str, Any]:
        ...

    def connect(self) -> bool:
        ...

    def close(self) -> None:
        ...


class RuntimeDriver(CommandDriver, QueryDriver, Protocol):
    pass


class BaseDriver(ABC):
    """Contract that every robot body driver must fulfil."""

    def __init__(self) -> None:
        self._state = DriverState.DISCONNECTED

    @property
    def state(self) -> DriverState:
        return self._state

    @abstractmethod
    def get_profile_path(self) -> Path | None:
        """Return an optional EMBODIED.md profile path."""

    @abstractmethod
    def load_environment(self, environment: dict[str, Any]) -> None:
        """Initialize driver state from an environment document."""

    def execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute one atomic action and return a structured result."""
        self.begin_action()
        try:
            result = self._execute_action(action_type, parameters)
        except Exception:
            self.mark_fault()
            raise
        self.finish_action(success=_result_success(result))
        return result

    @abstractmethod
    def _execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Driver-specific action implementation called under the state machine."""

    @abstractmethod
    def get_environment(self) -> dict[str, Any]:
        """Return the current environment document."""

    def connect(self) -> bool:
        """Establish a connection to the embodiment if needed."""
        if self._state == DriverState.CLOSED:
            raise DriverStateError("cannot connect a closed driver")
        self._state = DriverState.IDLE
        return True

    def disconnect(self) -> None:
        """Close the current connection if the driver maintains one."""
        if self._state != DriverState.CLOSED:
            self._state = DriverState.DISCONNECTED

    def is_connected(self) -> bool:
        """Return whether the driver currently considers itself connected."""
        return self._state in {DriverState.IDLE, DriverState.EXECUTING, DriverState.FAULT}

    def health_check(self) -> bool:
        """Run a lightweight connection health check."""
        return self.is_connected() and self._state != DriverState.FAULT

    def ensure_ready(self) -> None:
        if self._state != DriverState.IDLE:
            raise DriverStateError(f"driver is not ready: state={self._state.value}")

    def begin_action(self) -> None:
        self.ensure_ready()
        self._state = DriverState.EXECUTING

    def finish_action(self, *, success: bool = True) -> None:
        self._state = DriverState.IDLE if success else DriverState.FAULT

    def mark_fault(self) -> None:
        self._state = DriverState.FAULT

    def reset_fault(self) -> None:
        if self._state != DriverState.FAULT:
            raise DriverStateError(f"driver is not in fault state: state={self._state.value}")
        self._state = DriverState.IDLE

    def get_capabilities(self) -> dict[str, Any]:
        """Return supported actions and safety limits for validators/agents."""
        return {}

    def get_runtime_state(self) -> dict[str, Any]:
        """Return optional runtime state such as nav or connection status.

        Merged into ENVIRONMENT.md after each poll cycle so downstream
        consumers (Planner, tools) can inspect live state.
        """
        return {
            "connected": self.is_connected(),
            "healthy": self.health_check(),
            "driver_state": self._state.value,
            "capabilities": self.get_capabilities(),
        }

    def close(self) -> None:
        """Release hardware resources. Override if needed."""
        self._state = DriverState.CLOSED

    def __enter__(self) -> "BaseDriver":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _result_success(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    return result.get("success") is not False and not result.get("error")
