"""
hal/base_driver.py

Abstract base class for all robot body drivers.

Every hardware or simulation embodiment MUST subclass `BaseDriver` and
implement the four abstract methods. The Watchdog loads a driver by
name and interacts with it exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDriver(ABC):
    """Contract that every robot body driver must fulfil."""

    @abstractmethod
    def get_profile_path(self) -> Path | None:
        """Return an optional EMBODIED.md profile path."""

    @abstractmethod
    def load_environment(self, environment: dict[str, Any]) -> None:
        """Initialize driver state from an environment document."""

    @abstractmethod
    def execute_action(self, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute one atomic action and return a structured result."""

    @abstractmethod
    def get_environment(self) -> dict[str, Any]:
        """Return the current environment document."""

    def connect(self) -> bool:
        """Establish a connection to the embodiment if needed."""
        return True

    def disconnect(self) -> None:
        """Close the current connection if the driver maintains one."""

    def is_connected(self) -> bool:
        """Return whether the driver currently considers itself connected."""
        return True

    def health_check(self) -> bool:
        """Run a lightweight connection health check."""
        return self.is_connected()

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
            "capabilities": self.get_capabilities(),
        }

    def close(self) -> None:
        """Release hardware resources. Override if needed."""

    def __enter__(self) -> "BaseDriver":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
