import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.file_watcher import FileWatcher


def test_file_watcher_wakes_on_atomic_replace(tmp_path):
    watched = tmp_path / "ACTION.md"
    watched.write_text("before", encoding="utf-8")
    watcher = FileWatcher()

    def replace_file() -> None:
        time.sleep(0.1)
        tmp = tmp_path / ".ACTION.md.tmp"
        tmp.write_text("after", encoding="utf-8")
        tmp.replace(watched)

    thread = threading.Thread(target=replace_file)
    thread.start()
    try:
        changed = watcher.wait_for_change(watched, timeout=2.0)
    finally:
        watcher.close()
        thread.join()

    assert changed
    assert watched.read_text(encoding="utf-8") == "after"


def test_file_watcher_returns_false_on_timeout(tmp_path):
    watched = tmp_path / "ACTION.md"
    watched.write_text("still", encoding="utf-8")

    with FileWatcher() as watcher:
        changed = watcher.wait_for_change(watched, timeout=0.05)

    assert not changed
