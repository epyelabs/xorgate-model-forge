"""The labels file an `autolabel` job writes (`labels.json`), in the shape the
platform imports as `source: model` annotations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .common import ContractModel


class LabelBox(ContractModel):
    cls: str = Field(description="The dataset class after classMap.")
    modelCls: str = Field(description="The model's own class name.")
    confidence: float = Field(ge=0, le=1)
    xyxy: Annotated[list[float], Field(min_length=4, max_length=4)] = Field(description="Absolute pixels.")


class LabelsImage(ContractModel):
    id: str
    boxes: list[LabelBox]


class LabelsModel(ContractModel):
    family: str
    name: str
    version: int = Field(ge=1)


class LabelsSummary(ContractModel):
    images: int = Field(ge=0)
    boxes: int = Field(ge=0)
    perClass: dict[str, int] = Field(default_factory=dict)
    unmapped: dict[str, int] = Field(
        default_factory=dict, description="Model classes above minConfidence with no classMap entry."
    )


class LabelsFile(ContractModel):
    labelsVersion: Literal[1] = 1
    jobId: str
    model: LabelsModel
    minConfidence: float = Field(ge=0, le=1)
    classMap: dict[str, str] = Field(default_factory=dict)
    images: list[LabelsImage]
    summary: LabelsSummary
