from __future__ import annotations

import time
from typing import Any

from runtime.registry import Registry
from runtime.trace import TraceLogger
from tools.base import Tool
from tools.response import ToolResponse


class ToolRegistry(Registry[Tool]):
    def __init__(self, trace_logger: TraceLogger | None = None):
        super().__init__()
        self.trace_logger = trace_logger

    def register(self, tool: Tool) -> None:
        super().register(tool.name, tool)

    def get(self, name: str) -> Tool | None:
        return self._items.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self._items.values()]

    def run(self, name: str, parameters: dict[str, Any] | None = None) -> ToolResponse:
        params = parameters or {}
        if self.trace_logger is not None:
            self.trace_logger.log("tool_call", {"tool_name": name, "parameters": params})

        tool = self.get(name)
        if tool is None:
            response = ToolResponse.failure("tool_not_found", f"Tool {name!r} is not registered")
            if self.trace_logger is not None:
                self.trace_logger.log("tool_result", response.to_dict())
            return response

        start = time.time()
        try:
            response = tool.run(params)
        except Exception as exc:
            response = ToolResponse.failure(
                "tool_exception",
                f"Tool {name!r} raised {type(exc).__name__}: {exc}",
                context={"tool_name": name, "parameters": params},
            )
            if self.trace_logger is not None:
                self.trace_logger.log("tool_result", response.to_dict())
            return response

        stats = dict(response.stats)
        stats.setdefault("time_ms", int((time.time() - start) * 1000))
        context = dict(response.context)
        context.setdefault("tool_name", name)
        wrapped = ToolResponse(response.status, response.text, response.data, response.error, stats, context)
        if self.trace_logger is not None:
            self.trace_logger.log("tool_result", wrapped.to_dict())
        return wrapped
