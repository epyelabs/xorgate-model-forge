"""Progress v1 (`progress.jsonl`, and NDJSON on stdout with `--progress json`).

One object per line, `t: status | epoch | log`, each stamped `at`. The same
message shapes whatever the transport (a file synced to S3, stdout, later a
topic), as the engine's progress messages are."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from .common import ContractModel

Phase = Literal["prepare", "train", "export", "check", "evaluate", "label", "done"]


class StatusMessage(ContractModel):
    t: Literal["status"] = "status"
    at: str
    status: Literal["running", "succeeded", "failed"]
    phase: Phase


class EpochMessage(ContractModel):
    t: Literal["epoch"] = "epoch"
    at: str
    epoch: int = Field(ge=1)
    epochs: int = Field(ge=1)
    loss: float | None = None
    mAP50: float | None = Field(default=None, description="When the recipe validates per epoch.")


class LogMessage(ContractModel):
    t: Literal["log"] = "log"
    at: str
    line: str


ProgressMessage = Annotated[StatusMessage | EpochMessage | LogMessage, Field(discriminator="t")]
progress_adapter: TypeAdapter[ProgressMessage] = TypeAdapter(ProgressMessage)
