"""Result v1 (`result.json`, written last, always written even on failure)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import ContractModel
from .dataset import SplitCounts
from .job import BaseSpec
from .progress import Phase


class ResultDataset(ContractModel):
    classes: list[str]
    images: SplitCounts
    instances: dict[str, int] = Field(default_factory=dict, description="Ground-truth boxes per class, all splits.")


class EffectiveHyperparameters(ContractModel):
    """After the preset and auto batch."""

    epochs: int = Field(ge=1)
    imgsz: int = Field(ge=32)
    batch: int = Field(ge=1)
    patience: int = Field(ge=0)
    seed: int = Field(ge=0)


class PerClassMetrics(ContractModel):
    name: str
    mAP50: float
    mAP50_95: float | None = None
    precision: float
    recall: float
    instances: int = Field(ge=0, description="Ground-truth boxes in the validation split.")


class Metrics(ContractModel):
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    perClass: list[PerClassMetrics] = Field(default_factory=list)


class ResultModel(ContractModel):
    family: str
    name: str
    version: int = Field(ge=1)
    dir: str = Field(description="Relative to the job's output directory: model/<family>/<name>/<version>")


class CheckBlock(ContractModel):
    ok: bool
    lines: list[str] = Field(description="The four check lines, as `xmf check` prints them.")


class EvalArtifacts(ContractModel):
    metrics: str
    confusion: str
    samples: list[str] = Field(default_factory=list)


class Artifacts(ContractModel):
    trainLog: str | None = None
    weights: str | None = None
    eval: EvalArtifacts | None = None
    labels: str | None = None


class ResultLabels(ContractModel):
    file: str
    images: int = Field(ge=0)
    boxes: int = Field(ge=0)
    perClass: dict[str, int] = Field(default_factory=dict)
    unmapped: dict[str, int] = Field(default_factory=dict)


class JobError(ContractModel):
    phase: Phase
    message: str


class JobResult(ContractModel):
    resultVersion: Literal[1] = 1
    jobId: str
    task: Literal["train", "autolabel"]
    recipe: str
    status: Literal["succeeded", "failed"]
    startedAt: str
    endedAt: str
    durationMs: int = Field(ge=0)
    dataset: ResultDataset | None = None
    base: BaseSpec | None = None
    hyperparameters: EffectiveHyperparameters | None = None
    epochsCompleted: int | None = Field(default=None, ge=0)
    metrics: Metrics | None = None
    model: ResultModel | None = None
    check: CheckBlock | None = None
    artifacts: Artifacts = Field(default_factory=Artifacts)
    labels: ResultLabels | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    error: JobError | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = super().to_json_dict()
        data["error"] = self.error.to_json_dict() if self.error else None
        return data
