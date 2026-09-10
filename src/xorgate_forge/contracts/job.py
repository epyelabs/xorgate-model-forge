"""Job spec v1 (`job.json`, design 3b). Two tasks, discriminated on `task`."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from .common import ContractModel
from .manifest import MODEL_NAME_RE


class OutputSpec(ContractModel):
    """Where a `train` job publishes: the registry identity of the model."""

    family: str = Field(pattern=MODEL_NAME_RE, description='"detection"')
    name: str = Field(pattern=MODEL_NAME_RE, description="The registry name a node's config carries.")
    version: int = Field(ge=1)
    label: str | None = Field(default=None, description="Human name pickers show.")


class BaseSpec(ContractModel):
    """The weights training starts from."""

    kind: Literal["pretrained", "weights"]
    ref: str | None = Field(default=None, description='pretrained: a recipe base such as "yolov8n"')
    path: str | None = Field(default=None, description="weights: a path to a .pt file (train-from-previous)")

    @model_validator(mode="after")
    def _one_of(self) -> BaseSpec:
        if self.kind == "pretrained" and not self.ref:
            raise ValueError('base.kind "pretrained" needs base.ref')
        if self.kind == "weights" and not self.path:
            raise ValueError('base.kind "weights" needs base.path')
        return self

    def describe(self) -> str:
        return f"{self.kind}:{self.ref or self.path}"


class Hyperparameters(ContractModel):
    """Overrides on top of the recipe's preset (`mode`). Every field is
    optional here; the result carries the effective values."""

    epochs: int | None = Field(default=None, ge=1)
    imgsz: int | None = Field(default=None, ge=32)
    batch: Literal["auto"] | int | None = Field(default=None)
    patience: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _batch_positive(self) -> Hyperparameters:
        if isinstance(self.batch, int) and self.batch < 1:
            raise ValueError('hyperparameters.batch must be a positive integer or "auto"')
        return self


class Limits(ContractModel):
    maxMinutes: int = Field(default=60, ge=1, description="Wall-clock budget; training stops cleanly at it.")


class ModelRefSpec(ContractModel):
    """A registry model resolved through the model source."""

    family: str = Field(pattern=MODEL_NAME_RE)
    name: str = Field(description='"yolov8n" or "yolov8n@1" (a pinned version rides in the name)')


class _JobBase(ContractModel):
    specVersion: Literal[1] = 1
    jobId: str = Field(min_length=1)
    recipe: str = Field(min_length=1)
    limits: Limits = Field(default_factory=Limits)
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque; copied into the manifest and the result. The platform puts orgId, datasetVersionId here.",
    )


class TrainJobSpec(_JobBase):
    task: Literal["train"]
    output: OutputSpec
    base: BaseSpec
    mode: Literal["fast", "accurate"] = "fast"
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)


class AutolabelJobSpec(_JobBase):
    task: Literal["autolabel"]
    model: ModelRefSpec
    minConfidence: float = Field(default=0.4, ge=0, le=1)
    classMap: dict[str, str] = Field(
        default_factory=dict,
        description="model class -> dataset class; unmapped model classes are counted, not written",
    )
    images: Literal["all"] | list[str] = "all"


JobSpec = Annotated[TrainJobSpec | AutolabelJobSpec, Field(discriminator="task")]
job_spec_adapter: TypeAdapter[JobSpec] = TypeAdapter(JobSpec)


def parse_job_spec(data: Any) -> TrainJobSpec | AutolabelJobSpec:
    return job_spec_adapter.validate_python(data)
