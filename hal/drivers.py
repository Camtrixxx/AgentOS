"""
hal/drivers.py

Driver registry — maps driver names to classes so the Watchdog and tools
can load drivers dynamically instead of hardcoding imports.
"""

from __future__ import annotations

from typing import Any

from hal.base_driver import BaseDriver
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.registry import Registry


class DriverRegistry(Registry[type[BaseDriver]]):
    """Registry of driver classes keyed by driver name."""

    def __init__(self) -> None:
        super().__init__()
        self._init_builtins()

    def _init_builtins(self) -> None:
        self.register("fake_manipulation", FakeManipulationDriver)
        try:
            from hal.robosuite_driver import RobosuiteDriver
        except ImportError:
            return
        self.register("robosuite", RobosuiteDriver)

    def load_driver(self, name: str, **kwargs: Any) -> BaseDriver:
        cls = self.get(name)
        if cls is None:
            available = ", ".join(self.list_names())
            raise KeyError(f"Unknown driver {name!r}. Available: {available}")
        return cls(**kwargs)


# Module-level singleton for backward compatibility
driver_registry = DriverRegistry()


def register_driver(name: str, driver_cls: type[BaseDriver]) -> None:
    driver_registry.register(name, driver_cls)


def load_driver(name: str, **kwargs: Any) -> BaseDriver:
    return driver_registry.load_driver(name, **kwargs)


def list_drivers() -> list[str]:
    return driver_registry.list_names()
