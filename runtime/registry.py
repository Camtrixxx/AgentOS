from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic type-safe registry for named items."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        self._items[name] = item

    def get(self, name: str) -> T | None:
        return self._items.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
