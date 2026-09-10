"""Dataset v1 (design 3b): a directory with `dataset.json`,
`annotations.coco.json`, `splits.json` and `images/`.

The COCO file is the one canonical annotation file: category ids are 0-based
indexes into `dataset.json`'s `classes`, boxes are `[x, y, w, h]` in absolute
pixels, image ids are strings (the platform's item ids; an integer id from a
foreign COCO file is accepted and read as its decimal string)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .common import ContractModel, OpenModel

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

Splits = dict[str, Split]
splits_adapter: TypeAdapter[Splits] = TypeAdapter(Splits)


class SplitCounts(ContractModel):
    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)


class DatasetSource(OpenModel):
    """Where the dataset came from. Opaque to the forge; the platform writes
    its own ids here."""

    platform: str | None = None
    datasetId: str | None = None
    datasetName: str | None = None
    version: int | None = None
    createdAt: str | None = None


class DatasetHeader(ContractModel):
    """`dataset.json`."""

    datasetVersion: Literal[1] = 1
    task: Literal["object_detection"] = "object_detection"
    classes: list[str] = Field(min_length=1, description="Class names in category-id order (0-based).")
    categoryIds: Literal["zero-based"] = "zero-based"
    images: int = Field(ge=0)
    splits: SplitCounts
    source: DatasetSource | None = None

    @field_validator("classes")
    @classmethod
    def _unique_non_empty(cls, classes: list[str]) -> list[str]:
        if any(not c.strip() for c in classes):
            raise ValueError("class names must not be empty")
        if len(set(classes)) != len(classes):
            raise ValueError("class names must be unique")
        return classes


class CocoCategory(ContractModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1)


class CocoImage(ContractModel):
    id: str
    file_name: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    group: str | None = Field(default=None, description="Split-by-group key (a recording session bucket).")
    source: dict[str, Any] | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("image id must be a string or an integer")
        if isinstance(v, int):
            return str(v)
        return v


class CocoAnnotation(ContractModel):
    id: int
    image_id: str
    category_id: int = Field(ge=0)
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)] = Field(description="[x, y, w, h] absolute pixels")
    area: float | None = None
    iscrowd: int = 0
    source: str | None = Field(default=None, description='"human" or "model"')
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("image_id", mode="before")
    @classmethod
    def _image_id_to_str(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("image_id must be a string or an integer")
        if isinstance(v, int):
            return str(v)
        return v


class CocoFile(ContractModel):
    """`annotations.coco.json`."""

    info: dict[str, Any] | None = None
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]

    @model_validator(mode="after")
    def _ids_unique(self) -> CocoFile:
        image_ids = [i.id for i in self.images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image ids must be unique")
        ann_ids = [a.id for a in self.annotations]
        if len(set(ann_ids)) != len(ann_ids):
            raise ValueError("annotation ids must be unique")
        cat_ids = [c.id for c in self.categories]
        if len(set(cat_ids)) != len(cat_ids):
            raise ValueError("category ids must be unique")
        return self
