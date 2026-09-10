"""The forge's five seams (design 3c), mirroring the engine's four host seams
plus a model source. A host binds them; the runner reads and writes only
through them. `LocalHost` (hosts/local.py) is the reference implementation;
`SageMakerHost` (hosts/sagemaker.py) is a path mapping over it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import JobResult, ModelManifest, ProgressMessage


@dataclass(frozen=True)
class ResolvedModel:
    """A registry model found under a root: `<root>/<family>/<name>/<version>/`."""

    family: str
    name: str
    version: int
    manifest: ModelManifest
    dir: Path
    model_path: Path
    root: Path


class ModelNotFoundError(FileNotFoundError):
    pass


class WeightsNotFoundError(FileNotFoundError):
    pass


@runtime_checkable
class DatasetSource(Protocol):
    def resolve(self) -> Path:
        """A dataset v1 directory."""
        ...


@runtime_checkable
class ModelSource(Protocol):
    def fetch(self, family: str, name: str) -> ResolvedModel:
        """`name` may carry a pin as `name@version`; the newest version otherwise."""
        ...

    def fetch_weights(self, ref: str) -> Path:
        """Base weights for a recipe (`yolov8n` -> a `.pt` file)."""
        ...

    def describe(self) -> list[str]:
        """Where models and weights are looked for, for messages and `doctor`."""
        ...


@runtime_checkable
class ArtifactSink(Protocol):
    def dir(self) -> Path:
        """The job's output directory; everything the job writes lives under it."""
        ...

    def finalize(self, result: JobResult) -> Path:
        """Write `result.json` last. Returns its path."""
        ...


@runtime_checkable
class ProgressSink(Protocol):
    def emit(self, message: ProgressMessage) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class SecretReader(Protocol):
    def get(self, logical_name: str) -> str | None: ...


@dataclass
class Host:
    """What a bound host hands the runner."""

    name: str
    dataset: DatasetSource
    models: ModelSource
    artifacts: ArtifactSink
    progress: ProgressSink
    secrets: SecretReader
