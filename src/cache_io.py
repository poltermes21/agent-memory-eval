"""Atomic JSON cache read/write, shared by every arm and the judge.

These caches hold already-paid-for API results, and every runner saves after each
question so a crash loses at most one answer. But a plain path.write_text() is not
atomic: a process killed mid-write leaves a truncated or empty file, and the next
run then dies in json.load() -- blocking resume entirely and forcing a re-pay of
work that was already done. Happened 2026-08-12 (runs/arm_c/k20/conv-49.json, 0
bytes). Write to a temp file in the same directory, then os.replace(), which is
atomic on POSIX: readers see either the old complete file or the new complete one.
"""
import json
import os
import tempfile
from pathlib import Path


def load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
