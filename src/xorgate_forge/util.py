"""Small shared helpers: timestamps, hashing, byte formatting, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger("xmf")

LOG_LEVELS = {"silent": logging.CRITICAL + 10, "error": logging.ERROR, "info": logging.INFO, "debug": logging.DEBUG}


def configure_logging(level: str = "info") -> None:
    """stderr only; stdout is for results and the progress stream."""
    root = logging.getLogger("xmf")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[xmf] %(message)s"))
    root.addHandler(handler)
    root.setLevel(LOG_LEVELS.get(level, logging.INFO))
    root.propagate = False


def now_iso() -> str:
    """UTC, millisecond precision, `Z` suffix, like the engine's `at` stamps."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(UTC).microsecond // 1000:03d}Z"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(size: int) -> str:
    """The engine's `formatBytes` (src/local/models.ts), character for character."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{round(size / 1024):.0f} kB"
    return f"{size} B"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    """Write next to the target and rename, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
