from __future__ import annotations

from typing import Any, Protocol

from tools.response import ToolResponse


class Tool(Protocol):
    name: str
    description: str

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        ...

