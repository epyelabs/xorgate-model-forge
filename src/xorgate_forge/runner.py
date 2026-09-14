"""`xmf run`: the phases of a job over a bound host. `result.json` is
written last, always, even on failure (status `failed`, `error.phase` and
`error.message` set), so a finalizer never has to guess."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import __version__
from .assist import run_assist
from .check import check_model_dir
from .contracts import (
    Artifacts,
    AutolabelJobSpec,
    CheckBlock,
    EpochMessage,
    EvalArtifacts,
    JobError,
    JobResult,
    LogMessage,
    Metrics,
    ModelManifest,
    Phase,
    ProgressMessage,
    ResultDataset,
    ResultLabels,
    ResultModel,
    StatusMessage,
    TrainJobSpec,
    parse_job_spec,
)
from .dataset import Dataset, dataset_hash, load_dataset
from .dataset.validate import validate_loaded
from .evaluate import evaluate_model
from .recipes import Recipe, TrainContext, get_recipe
from .seams import Host
from .util import LOG, now_iso, read_json, sha256_of, write_json


class JobFailed(Exception):
    def __init__(self, phase: Phase, message: str) -> None:
        super().__init__(message)
        self.phase: Phase = phase
        self.message = message


class UsageError(Exception):
    """Exit 2: nothing ran."""


@dataclass
class RunOutcome:
    result: JobResult
    result_path: Path
    job_dir: Path

    @property
    def exit_code(self) -> int:
        return 0 if self.result.status == "succeeded" else 1


def load_spec(path: Path) -> TrainJobSpec | AutolabelJobSpec:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError) as err:
        raise UsageError(f"{path}: {err}") from err
    try:
        return parse_job_spec(data)
    except ValidationError as err:
        issue = err.errors()[0]
        loc = ".".join(str(p) for p in issue.get("loc", ())) or "(root)"
        raise UsageError(f"{path}: {loc}: {issue.get('msg')}") from err


def train_budget_seconds(max_minutes: int) -> float:
    """Training gets the budget minus a reserve for export, check and
    evaluate: 20 percent, at most five minutes, at least thirty seconds."""
    total = max_minutes * 60.0
    reserve = min(300.0, total * 0.2)
    return max(30.0, total - reserve)


class JobRunner:
    def __init__(self, spec: TrainJobSpec | AutolabelJobSpec, host: Host, *, spec_path: Path | None = None) -> None:
        self.spec = spec
        self.host = host
        self.spec_path = spec_path
        self.job_dir = host.artifacts.dir()
        self.started_at = now_iso()
        self._t0 = time.monotonic()
        self.artifacts = Artifacts()
        self.phase: Phase = "prepare"

    # --- progress ---------------------------------------------------------------
    def emit(self, message: ProgressMessage) -> None:
        self.host.progress.emit(message)

    def status(self, phase: Phase, status: str = "running") -> None:
        self.phase = phase
        self.emit(StatusMessage(at=now_iso(), status=status, phase=phase))  # type: ignore[arg-type]
        LOG.info(f"{status}: {phase}")

    def log(self, line: str) -> None:
        self.emit(LogMessage(at=now_iso(), line=line))
        LOG.info(line)

    # --- run --------------------------------------------------------------------
    def run(self) -> RunOutcome:
        fields: dict[str, Any] = {}
        error: JobError | None = None
        try:
            self.status("prepare")
            if isinstance(self.spec, TrainJobSpec):
                fields = self._run_train(self.spec)
            else:
                fields = self._run_autolabel(self.spec)
            status = "succeeded"
        except JobFailed as err:
            status = "failed"
            error = JobError(phase=err.phase, message=err.message)
            LOG.error(f"{err.phase}: {err.message}")
        except Exception as err:
            status = "failed"
            error = JobError(phase=self.phase, message=f"{type(err).__name__}: {err}")
            LOG.error(f"{self.phase}: {type(err).__name__}: {err}")
            LOG.debug(traceback.format_exc())
        ended_at = now_iso()
        result = JobResult(
            jobId=self.spec.jobId,
            task=self.spec.task,
            recipe=self.spec.recipe,
            status=status,  # type: ignore[arg-type]
            startedAt=self.started_at,
            endedAt=ended_at,
            durationMs=int((time.monotonic() - self._t0) * 1000),
            artifacts=self.artifacts,
            provenance=dict(self.spec.provenance),
            error=error,
            **fields,
        )
        self.emit(StatusMessage(at=ended_at, status=status, phase="done"))  # type: ignore[arg-type]
        self.host.progress.close()
        path = self.host.artifacts.finalize(result)
        LOG.info(f"{status}: result.json -> {path}")
        return RunOutcome(result, path, self.job_dir)

    # --- shared prepare ---------------------------------------------------------
    def _load_dataset(self, *, require_splits: bool, require_val: bool) -> Dataset:
        root = self.host.dataset.resolve()
        try:
            ds = load_dataset(root, require_splits=require_splits)
        except (OSError, ValueError, ValidationError) as err:
            raise JobFailed("prepare", f"dataset {root}: {err}") from err
        report = validate_loaded(ds, open_images=False, require_val=require_val)
        for w in report.warnings:
            self.log(f"dataset warning: {w}")
        if report.errors:
            head = "; ".join(report.errors[:5])
            more = f" (+{len(report.errors) - 5} more)" if len(report.errors) > 5 else ""
            raise JobFailed("prepare", f"dataset {root} is not valid: {head}{more}")
        return ds

    def _recipe(self, recipe_id: str) -> Recipe:
        try:
            return get_recipe(recipe_id)
        except KeyError as err:
            raise JobFailed("prepare", str(err)) from err

    # --- train ------------------------------------------------------------------
    def _run_train(self, spec: TrainJobSpec) -> dict[str, Any]:
        recipe = self._recipe(spec.recipe)
        ds = self._load_dataset(require_splits=True, require_val=True)
        counts = ds.count_splits()
        instances: dict[str, int] = {c: 0 for c in ds.classes}
        cats = {c.id: c.name for c in ds.coco.categories}
        for ann in ds.coco.annotations:
            instances[cats[ann.category_id]] = instances.get(cats[ann.category_id], 0) + 1
        fields: dict[str, Any] = {
            "dataset": ResultDataset(classes=list(ds.classes), images=counts, instances=instances),
            "base": spec.base,
        }
        if spec.base.kind == "pretrained":
            assert spec.base.ref is not None
            if spec.base.ref not in recipe.base_refs:
                raise JobFailed(
                    "prepare", f'base "{spec.base.ref}" is not one of the {recipe.id} bases {list(recipe.base_refs)}'
                )
            try:
                weights = self.host.models.fetch_weights(spec.base.ref)
            except (OSError, ValueError) as err:
                raise JobFailed("prepare", str(err)) from err
        else:
            assert spec.base.path is not None
            weights = Path(spec.base.path)
            if not weights.is_absolute() and self.spec_path is not None:
                weights = (self.spec_path.parent / weights).resolve()
            if not weights.is_file():
                raise JobFailed("prepare", f"base weights {weights} not found")
        hp = recipe.resolve_hyperparameters(spec)
        self.log(
            f"recipe {recipe.id} ({recipe.license}), base {spec.base.describe()}, mode {spec.mode}: "
            + ", ".join(f"{k} {v}" for k, v in hp.items())
        )
        work_dir = Path(tempfile.mkdtemp(prefix=f"xmf-{spec.jobId}-"))
        try:
            self.status("train")
            budget = train_budget_seconds(spec.limits.maxMinutes)
            ctx = TrainContext(
                dataset=ds,
                spec=spec,
                host=self.host,
                job_dir=self.job_dir,
                work_dir=work_dir,
                weights=weights,
                hyperparameters=hp,
                train_budget_seconds=budget,
                on_epoch=lambda epoch, epochs, loss, map50: self.emit(
                    EpochMessage(at=now_iso(), epoch=epoch, epochs=epochs, loss=loss, mAP50=map50)
                ),
                on_log=self.log,
            )
            outcome = recipe.train(ctx)
            self.artifacts.trainLog = str(outcome.train_log.relative_to(self.job_dir))
            self.artifacts.weights = str(outcome.weights.relative_to(self.job_dir))
            fields["hyperparameters"] = outcome.effective
            fields["epochsCompleted"] = outcome.epochs_completed
            self.log(f"training done: {outcome.epochs_completed} epoch(s), best weights {self.artifacts.weights}")

            self.status("export")
            out = spec.output
            model_rel = Path("model") / out.family / out.name / str(out.version)
            dest = self.job_dir / model_rel
            if dest.exists():
                shutil.rmtree(dest)
            exported = recipe.export(outcome.weights, dest, imgsz=outcome.effective.imgsz, work_dir=work_dir)
            if exported.class_names != ds.classes:
                raise JobFailed(
                    "export", f"exported class names {exported.class_names} do not match the dataset's {ds.classes}"
                )
            manifest = self._build_manifest(spec, recipe, exported, ds, outcome.epochs_completed)
            # The files block as tools/export/export_yolo.py writes it, so the check verifies rather than fills.
            manifest["files"] = {
                "model.onnx": {
                    "sha256": sha256_of(exported.model_path),
                    "sizeBytes": exported.model_path.stat().st_size,
                }
            }
            write_json(dest / "manifest.json", manifest)
            fields["model"] = ResultModel(family=out.family, name=out.name, version=out.version, dir=str(model_rel))
            self.log(
                f"exported {out.family}/{out.name} v{out.version} -> {model_rel} ({exported.imgsz}x{exported.imgsz})"
            )

            self.status("check")
            report = check_model_dir(dest, fill=True)
            fields["check"] = CheckBlock(ok=report.ok, lines=report.rendered())
            for line in report.rendered():
                self.log(line)
            if not report.ok:
                raise JobFailed("check", "; ".join(report.problems))

            self.status("evaluate")
            parsed = ModelManifest.model_validate(read_json(dest / "manifest.json"))
            ev = evaluate_model(dest, parsed, ds, self.job_dir / "eval")
            fields["metrics"] = ev.metrics
            self.artifacts.eval = EvalArtifacts(
                metrics=str(ev.metrics_file.relative_to(self.job_dir)),
                confusion=str(ev.confusion_file.relative_to(self.job_dir)),
                samples=[str(p.relative_to(self.job_dir)) for p in ev.samples],
            )
            self.log(
                f"evaluated on {ev.val_images} val image(s): mAP50 {ev.metrics.mAP50:.3f}, "
                f"mAP50-95 {ev.metrics.mAP50_95:.3f}, "
                f"precision {ev.metrics.precision:.3f}, recall {ev.metrics.recall:.3f} at confidence {ev.confidence}"
            )
            # The manifest's provenance carries the metrics of the artifact it describes.
            manifest["provenance"]["metrics"] = {
                "mAP50": ev.metrics.mAP50,
                "mAP50_95": ev.metrics.mAP50_95,
                "confidence": ev.metrics.confidence,
            }
            write_json(dest / "manifest.json", manifest)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        return fields

    def _build_manifest(
        self, spec: TrainJobSpec, recipe: Recipe, exported: Any, ds: Dataset, epochs_completed: int
    ) -> dict[str, Any]:
        """Manifest v1 as `tools/export/export_yolo.py` writes it, plus `provenance`."""
        out = spec.output
        platform = dict(spec.provenance)
        return {
            "manifestVersion": 1,
            "family": out.family,
            "name": out.name,
            "version": out.version,
            "label": out.label or out.name,
            "backend": "cpu-onnx",
            "input": {
                "width": exported.imgsz,
                "height": exported.imgsz,
                "layout": "nchw",
                "colorSpace": "rgb",
                "normalize": {"scale": 1.0 / 255.0},
                "letterbox": {"enabled": True, "padValue": 114},
            },
            "classNames": exported.class_names,
            "postprocess": exported.postprocess,
            "quantization": "fp32",
            "provenance": {
                "trainedBy": f"xorgate-model-forge@{__version__}",
                "recipe": recipe.id,
                "recipeLicense": recipe.license,
                "base": spec.base.to_json_dict(),
                "job": spec.jobId,
                "dataset": {
                    "id": platform.get("datasetId"),
                    "version": platform.get("datasetVersion"),
                    "images": len(ds.coco.images),
                    "hash": dataset_hash(ds.root),
                },
                "metrics": {},
                "trainedAt": now_iso(),
                "epochsCompleted": epochs_completed,
                "platform": platform,
            },
        }

    # --- autolabel --------------------------------------------------------------
    def _run_autolabel(self, spec: AutolabelJobSpec) -> dict[str, Any]:
        self._recipe(spec.recipe)
        ds = self._load_dataset(require_splits=False, require_val=False)
        missing = sorted(set(spec.classMap.values()) - set(ds.classes))
        if missing:
            raise JobFailed("prepare", f"classMap names class(es) the dataset lacks: {missing} (classes: {ds.classes})")
        if isinstance(spec.images, list):
            known = {i.id for i in ds.coco.images}
            unknown = [i for i in spec.images if i not in known]
            if unknown:
                raise JobFailed("prepare", f"images not in the dataset: {unknown[:5]}")
        try:
            model = self.host.models.fetch(spec.model.family, spec.model.name)
        except (OSError, ValueError, ValidationError) as err:
            raise JobFailed("prepare", str(err)) from err
        if model.manifest.postprocess.kind != "yolov8":
            raise JobFailed(
                "prepare",
                f"model {model.family}/{model.name}: "
                f'postprocess.kind "{model.manifest.postprocess.kind}" has no decoder in the forge',
            )
        self.log(
            f"assist with {model.family}/{model.name} v{model.version} ({len(model.manifest.classNames)} classes), "
            f"minConfidence {spec.minConfidence}, classMap {spec.classMap or 'identity'}"
        )
        self.status("label")
        labels_file = self.job_dir / "labels.json"
        labels = run_assist(spec, ds, model, labels_file)
        self.artifacts.labels = "labels.json"
        s = labels.summary
        self.log(f"labeled {s.images} image(s): {s.boxes} box(es) {s.perClass}, unmapped {s.unmapped}")
        return {
            "dataset": ResultDataset(classes=list(ds.classes), images=ds.count_splits(), instances={}),
            "labels": ResultLabels(
                file="labels.json", images=s.images, boxes=s.boxes, perClass=s.perClass, unmapped=s.unmapped
            ),
        }


def run_job(spec: TrainJobSpec | AutolabelJobSpec, host: Host, *, spec_path: Path | None = None) -> RunOutcome:
    return JobRunner(spec, host, spec_path=spec_path).run()


__all__ = [
    "JobFailed",
    "JobRunner",
    "Metrics",
    "RunOutcome",
    "UsageError",
    "load_spec",
    "run_job",
    "train_budget_seconds",
]
