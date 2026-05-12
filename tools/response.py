from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True)
class ToolResponse:
    status: ToolStatus
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        text: str,
        *,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> "ToolResponse":
        return cls(ToolStatus.SUCCESS, text, data or {}, None, stats or {}, context or {})

    @classmethod
    def partial(
        cls,
        text: str,
        *,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> "ToolResponse":
        return cls(ToolStatus.PARTIAL, text, data or {}, None, stats or {}, context or {})

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> "ToolResponse":
        return cls(
            ToolStatus.ERROR,
            message,
            data or {},
            {"code": code, "message": message},
            stats or {},
            context or {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status.value, "text": self.text, "data": self.data}
        if self.error is not None:
            payload["error"] = self.error
        if self.stats:
            payload["stats"] = self.stats
        if self.context:
            payload["context"] = self.context
        return payload

