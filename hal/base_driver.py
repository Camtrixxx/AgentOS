from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDriver(ABC):
    """Minimal HAL contract for fake and future physical embodiments."""

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
        return True

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def health_check(self) -> bool:
        return self.is_connected()

    def close(self) -> None:
        return None

    def __enter__(self) -> "BaseDriver":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

