from __future__ import annotations

from pathlib import Path

from runtime.environment_io import utc_now_iso


def append_lesson(path: Path, *, title: str, details: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Lessons\n"
    addition = f"\n## {utc_now_iso()} - {title}\n\n{details.strip()}\n"
    path.write_text(existing.rstrip() + "\n" + addition, encoding="utf-8")

