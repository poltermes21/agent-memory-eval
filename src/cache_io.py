"""Atomic JSON cache read/write, shared by every arm and the judge.

These hold already-paid-for API results. write_text() is not atomic: a process
killed mid-write leaves a truncated file that blocks the next resume and forces
a re-pay. Write to a temp file, then os.replace().
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
