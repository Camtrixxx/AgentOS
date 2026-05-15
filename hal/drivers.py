"""
hal/drivers.py

Driver registry — maps driver names to classes so the Watchdog and tools
can load drivers dynamically instead of hardcoding imports.
"""

from __future__ import annotations

from typing import Any

from hal.base_driver import BaseDriver
from hal.fake_manipulation_driver import FakeManipulationDriver

_registry: dict[str, type[BaseDriver]] = {}


def _init_registry() -> None:
    if _registry:
        return
    _registry["fake_manipulation"] = FakeManipulationDriver


def register_driver(name: str, driver_cls: type[BaseDriver]) -> None:
    _init_registry()
    _registry[name] = driver_cls


def load_driver(name: str, **kwargs: Any) -> BaseDriver:
    _init_registry()
    cls = _registry.get(name)
    if cls is None:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown driver {name!r}. Available: {available}")
    return cls(**kwargs)


def list_drivers() -> list[str]:
    _init_registry()
    return sorted(_registry.keys())
