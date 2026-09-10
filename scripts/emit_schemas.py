#!/usr/bin/env python3
"""Emit the JSON Schemas for the forge's contracts into contracts/ from the
pydantic models (`make schemas`). The golden suite re-emits them into a
temp dir and fails when the committed copies drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from xorgate_forge import __version__
from xorgate_forge.contracts import (
    CocoFile,
    DatasetHeader,
    JobResult,
    LabelsFile,
    ModelManifest,
    job_spec_adapter,
    progress_adapter,
    splits_adapter,
)

SCHEMAS: dict[str, tuple[str, Any]] = {
    "dataset.schema.json": ("dataset v1: dataset.json", TypeAdapter(DatasetHeader)),
    "annotations.schema.json": ("dataset v1: annotations.coco.json", TypeAdapter(CocoFile)),
    "splits.schema.json": ("dataset v1: splits.json", splits_adapter),
    "job.schema.json": ("job spec v1: job.json", job_spec_adapter),
    "result.schema.json": ("result v1: result.json", TypeAdapter(JobResult)),
    "progress.schema.json": ("progress v1: one line of progress.jsonl", progress_adapter),
    "labels.schema.json": ("labels v1: the autolabel output", TypeAdapter(LabelsFile)),
    "manifest-with-provenance.schema.json": (
        "model manifest v1 with the forge's additive provenance block",
        TypeAdapter(ModelManifest),
    ),
}


def emit(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, (title, adapter) in SCHEMAS.items():
        schema = adapter.json_schema(mode="validation")
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://docs.xorgate.io/schemas/forge/{name}",
            "title": title,
            "x-forgeVersion": __version__,
            **schema,
        }
        path = out_dir / name
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "contracts"
    for p in emit(target):
        print(p)
