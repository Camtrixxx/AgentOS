from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime.environment_io import to_jsonable


class TraceLogger:
    """Small JSONL trace logger for embodied runtime events."""

    def __init__(self, output_dir: str | Path = "outputs/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = f"s-{ts}-{uuid.uuid4().hex[:4]}"
        self.path = self.output_dir / f"trace-{self.session_id}.jsonl"

    def log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": event,
            "payload": to_jsonable(payload or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

