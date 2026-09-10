"""A recipe is the unit of "which trainer" (design 3d): id, family, task,
license, presets, and `train` / `export`. Evaluate and assist are generic
over the exported ONNX and the manifest (evaluate.py, assist.py), so a
recipe only has to produce a registry directory the engine decodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..contracts import EffectiveHyperparameters, TrainJobSpec
from ..dataset import Dataset
from ..seams import Host


@dataclass(frozen=True)
class Preset:
    epochs: int
    imgsz: int
    batch: int | Literal["auto"]
    patience: int
    seed: int = 0


@dataclass
class TrainContext:
    dataset: Dataset
    spec: TrainJobSpec
    host: Host
    job_dir: Path
    work_dir: Path
    weights: Path
    hyperparameters: dict[str, Any]
    """Preset plus overrides; `batch` may still be "auto"."""
    train_budget_seconds: float
    on_epoch: Callable[[int, int, float | None, float | None], None]
    on_log: Callable[[str], None]


@dataclass
class TrainOutcome:
    weights: Path
    epochs_completed: int
    effective: EffectiveHyperparameters
    train_log: Path
    stop_reason: str | None = None


@dataclass
class ExportOutcome:
    model_path: Path
    class_names: list[str]
    imgsz: int
    postprocess: dict[str, Any]


class Recipe(ABC):
    id: str
    family: str
    task: str
    license: str
    base_refs: tuple[str, ...]
    presets: dict[str, Preset]
    weights_suffix: str = ".pt"

    def resolve_hyperparameters(self, spec: TrainJobSpec) -> dict[str, Any]:
        preset = self.presets[spec.mode]
        values: dict[str, Any] = {
            "epochs": preset.epochs,
            "imgsz": preset.imgsz,
            "batch": preset.batch,
            "patience": preset.patience,
            "seed": preset.seed,
        }
        for key, value in spec.hyperparameters.model_dump(exclude_none=True).items():
            values[key] = value
        return values

    @abstractmethod
    def train(self, ctx: TrainContext) -> TrainOutcome: ...

    @abstractmethod
    def export(self, weights: Path, dest: Path, *, imgsz: int, work_dir: Path) -> ExportOutcome: ...

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "task": self.task,
            "license": self.license,
            "baseWeights": list(self.base_refs),
            "presets": {k: v.__dict__ for k, v in self.presets.items()},
        }
