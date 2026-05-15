from __future__ import annotations

import os
import select
import time
from pathlib import Path


class FileWatcher:
    """Wait for file changes, using inotify when available and mtime polling otherwise."""

    def __init__(self) -> None:
        self._inotify_fd: int | None = None
        self._watch_descriptors: dict[Path, int] = {}
        try:
            self._inotify_fd = os.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            self._inotify_fd = None

    def wait_for_change(self, path: str | Path, *, timeout: float) -> bool:
        watched_path = Path(path)
        if self._inotify_fd is not None:
            return self._wait_inotify(watched_path, timeout=timeout)
        return self._wait_mtime(watched_path, timeout=timeout)

    def close(self) -> None:
        if self._inotify_fd is not None:
            os.close(self._inotify_fd)
            self._inotify_fd = None
            self._watch_descriptors.clear()

    def __enter__(self) -> "FileWatcher":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _wait_inotify(self, path: Path, *, timeout: float) -> bool:
        assert self._inotify_fd is not None
        watch_target = path.parent if path.parent != Path("") else Path(".")
        watch_target.mkdir(parents=True, exist_ok=True)
        resolved_target = watch_target.resolve()
        if resolved_target not in self._watch_descriptors:
            mask = _inotify_mask()
            wd = os.inotify_add_watch(self._inotify_fd, str(resolved_target), mask)  # type: ignore[attr-defined]
            self._watch_descriptors[resolved_target] = wd

        readable, _, _ = select.select([self._inotify_fd], [], [], max(timeout, 0.0))
        if not readable:
            return False
        try:
            os.read(self._inotify_fd, 4096)
        except BlockingIOError:
            return False
        return True

    def _wait_mtime(self, path: Path, *, timeout: float) -> bool:
        deadline = time.monotonic() + max(timeout, 0.0)
        initial = _mtime_ns(path)
        while time.monotonic() < deadline:
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
            if _mtime_ns(path) != initial:
                return True
        return False


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _inotify_mask() -> int:
    return (
        getattr(os, "IN_CLOSE_WRITE", 0x00000008)
        | getattr(os, "IN_MOVED_TO", 0x00000080)
        | getattr(os, "IN_CREATE", 0x00000100)
        | getattr(os, "IN_DELETE", 0x00000200)
        | getattr(os, "IN_ATTRIB", 0x00000004)
    )
