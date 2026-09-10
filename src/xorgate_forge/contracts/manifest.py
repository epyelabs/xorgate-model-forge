"""Model manifest v1 as the engine defines it (`modelManifestSchema`,
`src/models/registry.ts`), plus the additive `provenance` block the forge
writes (design 3b, Q10). The engine ignores `provenance` on parse today and
keeps it in the file; `contracts/model-manifest.schema.json` is the engine's
own JSON Schema, vendored, and the golden suite asserts the copy is current."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .common import ContractModel, OpenModel

# The engine's segment rule (src/local/model-ref.ts SEGMENT_RE) for families and names.
MODEL_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class Normalize(ContractModel):
    scale: float = Field(default=1 / 255, gt=0)
    mean: Vec3 | None = None
    std: Vec3 | None = None


class Letterbox(ContractModel):
    enabled: bool = True
    padValue: int = Field(default=114, ge=0, le=255)


class ManifestInput(ContractModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    layout: Literal["nchw"] = "nchw"
    colorSpace: Literal["rgb"] = "rgb"
    normalize: Normalize = Field(default_factory=Normalize)
    letterbox: Letterbox = Field(default_factory=Letterbox)


class ManifestPostprocess(ContractModel):
    kind: Literal["yolov8", "rtdetr", "yolov8-seg", "depth-anything"]
    nmsIouThreshold: float = Field(default=0.45, ge=0, le=1)
    maskThreshold: float = Field(default=0.5, ge=0, le=1)


class FileIntegrity(ContractModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sizeBytes: int = Field(ge=0)


class ProvenanceBase(ContractModel):
    kind: Literal["pretrained", "weights"]
    ref: str | None = None
    path: str | None = None


class ProvenanceDataset(ContractModel):
    id: str | None = None
    version: int | None = None
    images: int = Field(ge=0)
    hash: str | None = Field(
        default=None, description="sha256 over dataset.json, annotations.coco.json and splits.json"
    )


class ProvenanceMetrics(ContractModel):
    mAP50: float | None = None
    mAP50_95: float | None = None


class Provenance(ContractModel):
    trainedBy: str = Field(description='"xorgate-model-forge@<version>"')
    recipe: str
    recipeLicense: str
    base: ProvenanceBase
    job: str
    dataset: ProvenanceDataset
    metrics: ProvenanceMetrics
    trainedAt: str
    platform: dict[str, Any] = Field(default_factory=dict, description="The job spec's opaque provenance block.")


class ModelManifest(ContractModel):
    manifestVersion: Literal[1] = 1
    family: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    label: str | None = Field(default=None, min_length=1)
    backend: Literal["cpu-onnx", "gpu-endpoint"]
    input: ManifestInput
    classNames: list[str] = Field(default_factory=list)
    postprocess: ManifestPostprocess
    quantization: Literal["fp32", "fp16", "int8"] | None = None
    files: dict[str, FileIntegrity] | None = None
    provenance: Provenance | None = Field(
        default=None, description="Additive; written by the forge, ignored by engine 0.3.x."
    )


__all__ = [
    "MODEL_NAME_RE",
    "FileIntegrity",
    "Letterbox",
    "ManifestInput",
    "ManifestPostprocess",
    "ModelManifest",
    "Normalize",
    "OpenModel",
    "Provenance",
    "ProvenanceBase",
    "ProvenanceDataset",
    "ProvenanceMetrics",
]
