#!/usr/bin/env python3
"""Pull the large golden fixtures listed in fixtures/fixtures.json (the
engine's drive.mp4 and the yolov8n model), verifying sha256 and size. A
no-op when every file is present and matches. XMF_FIXTURES_URL overrides
the base URL (http(s): or file:) for a mirror. An entry already present in
the engine checkout next to this repo, or in ~/.xorgate/models, is copied
from there instead of downloaded."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
LOCAL_SOURCES = [
    ROOT.parent / "xorgate-workflow-engine" / "fixtures",
    Path.home() / ".xorgate",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matches(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["sizeBytes"] and sha256_of(path) == entry["sha256"]


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if url.startswith("file:"):
        shutil.copyfile(
            url[len("file:") :].lstrip("/") if not url.startswith("file:///") else url[len("file://") :], tmp
        )
    else:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    tmp.replace(target)


def main() -> int:
    spec = json.loads((FIXTURES / "fixtures.json").read_text())
    base = os.environ.get("XMF_FIXTURES_URL") or spec["baseUrl"]
    for rel, entry in spec["files"].items():
        target = FIXTURES / rel
        if matches(target, entry):
            print(f"ok       {rel}")
            continue
        found = None
        for source_root in LOCAL_SOURCES:
            candidate = source_root / rel
            if matches(candidate, entry):
                found = candidate
                break
        if found is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, target)
            print(f"copied   {rel} <- {found}")
        else:
            url = f"{base.rstrip('/')}/{entry['url']}"
            print(f"pulling  {rel} <- {url}")
            fetch(url, target)
        if not matches(target, entry):
            print(f"FAIL     {rel}: sha256 or size mismatch after fetch", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
