"""The `yolov8` recipe: Ultralytics YOLOv8 (code and weights under
AGPL-3.0; `xmf recipes` and the docs say so, and the license decision before
customer exposure is the owner's, plan Q2).

Train materializes a YOLO layout in a temp dir from the dataset v1
directory, runs the Ultralytics trainer with per-epoch callbacks that
become progress lines, stops early by `patience`, and honours the job's
time budget by asking the trainer to stop between batches. Export folds in
the engine's `tools/export/export_yolo.py`: ONNX at opset 12, static
`imgsz`, `postprocess.kind: yolov8`, which the engine decodes with zero
engine changes."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..contracts import EffectiveHyperparameters
from ..dataset.formats import export_yolo
from ..util import LOG
from .base import ExportOutcome, Preset, Recipe, TrainContext, TrainOutcome

NMS_IOU = 0.45
DEFAULT_CPU_BATCH = 16
# The AMP check in Ultralytics downloads a small model on CUDA hosts; when
# it is baked next to the base weights, it is copied into the trainer's cwd
# so the check finds it without a network.
AMP_CHECK_WEIGHTS = ("yolo26n.pt", "yolo11n.pt")


def _prepare_environment(work_dir: Path) -> None:
    """Keep Ultralytics inside the job's work dir and off the network for
    settings, fonts and telemetry (the image runs with network isolation)."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(work_dir / "ultralytics-config"))
    os.environ.setdefault("YOLO_VERBOSE", "False")
    (work_dir / "ultralytics-config").mkdir(parents=True, exist_ok=True)


def _route_ultralytics_logging(train_log: Path, level: int) -> logging.FileHandler:
    """Ultralytics logs to stdout by default; stdout is the progress
    stream, so its lines go to `train.log` and, at the CLI's level, stderr."""
    import ultralytics.utils as uu

    logger = uu.LOGGER
    for h in list(logger.handlers):
        logger.removeHandler(h)
    file_handler = logging.FileHandler(train_log, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(logging.Formatter("[ultralytics] %(message)s"))
    err.setLevel(level)
    logger.addHandler(err)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return file_handler


def download_ultralytics_asset(target: Path) -> None:
    """Fetch a release asset (`yolov8n.pt`) into `target`; the progress bar
    is off and stdout is redirected so nothing reaches the progress stream."""
    from ultralytics.utils.downloads import attempt_download_asset

    with contextlib.redirect_stdout(sys.stderr):
        attempt_download_asset(str(target), progress=False)


def select_device() -> str:
    forced = os.environ.get("XMF_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class Yolov8Recipe(Recipe):
    id = "yolov8"
    family = "detection"
    task = "object_detection"
    license = "AGPL-3.0"
    base_refs = ("yolov8n", "yolov8s")
    presets = {
        "fast": Preset(epochs=50, imgsz=640, batch="auto", patience=10, seed=0),
        "accurate": Preset(epochs=150, imgsz=640, batch="auto", patience=30, seed=0),
    }

    def train(self, ctx: TrainContext) -> TrainOutcome:
        _prepare_environment(ctx.work_dir)
        train_log = ctx.job_dir / "train.log"
        stderr_level = logging.INFO if LOG.isEnabledFor(logging.DEBUG) else logging.WARNING
        file_handler = _route_ultralytics_logging(train_log, stderr_level)
        from ultralytics import YOLO  # type: ignore[attr-defined]

        layout = ctx.work_dir / "yolo"
        data_yaml = export_yolo(ctx.dataset, layout, link=True)
        counts = ctx.dataset.count_splits()
        ctx.on_log(
            f"materialized yolo layout: {counts.train} train, {counts.val} val images, "
            f"{len(ctx.dataset.classes)} classes"
        )
        device = select_device()
        hp = ctx.hyperparameters
        batch = hp["batch"]
        if batch == "auto":
            batch = -1 if device != "cpu" else DEFAULT_CPU_BATCH
        workers = int(os.environ.get("XMF_WORKERS", "4"))
        for name in AMP_CHECK_WEIGHTS:
            baked = ctx.weights.parent / name
            if baked.is_file() and not (ctx.work_dir / name).exists():
                shutil.copy2(baked, ctx.work_dir / name)

        started = time.monotonic()
        deadline = started + ctx.train_budget_seconds
        state: dict[str, Any] = {"epochs_completed": 0, "stop_reason": None, "batch": batch, "epochs": hp["epochs"]}

        def on_train_batch_end(trainer: Any) -> None:
            if time.monotonic() > deadline and not trainer.stop:
                trainer.stop = True
                state["stop_reason"] = (
                    f"time budget: {ctx.train_budget_seconds / 60:.1f} min reached during epoch {trainer.epoch + 1}"
                )
                ctx.on_log(str(state["stop_reason"]))

        def on_train_epoch_end(trainer: Any) -> None:
            # Once per trained epoch (also for one cut short by the budget), before its validation.
            state["epochs_completed"] = trainer.epoch + 1

        def on_fit_epoch_end(trainer: Any) -> None:
            # final_eval() fires this once more with the best weights' metrics and epoch + 1; not an epoch.
            if trainer.epoch + 1 != state["epochs_completed"] or state.get("last_reported") == trainer.epoch + 1:
                return
            state["last_reported"] = trainer.epoch + 1
            state["batch"] = int(getattr(trainer, "batch_size", batch) or batch)
            state["epochs"] = int(trainer.epochs)
            loss: float | None = None
            tloss = getattr(trainer, "tloss", None)
            with contextlib.suppress(Exception):
                if tloss is not None:
                    values = list(tloss.values()) if hasattr(tloss, "values") else list(tloss)
                    loss = float(sum(float(v) for v in values))
            metrics = getattr(trainer, "metrics", None) or {}
            map50 = metrics.get("metrics/mAP50(B)")
            ctx.on_epoch(trainer.epoch + 1, int(trainer.epochs), loss, float(map50) if map50 is not None else None)
            if trainer.stop and state["stop_reason"] is None and trainer.epoch + 1 < trainer.epochs:
                state["stop_reason"] = (
                    f"early stop: no improvement for {trainer.args.patience} epochs "
                    f"(best epoch {trainer.stopper.best_epoch + 1})"
                )
                ctx.on_log(str(state["stop_reason"]))

        model = YOLO(str(ctx.weights))
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        project = ctx.work_dir / "runs"
        real_stdout = sys.stdout
        try:
            # Anything the trainer prints (progress bars, tables) must not reach the progress stream.
            with contextlib.redirect_stdout(sys.stderr if LOG.isEnabledFor(logging.DEBUG) else _Devnull()):
                model.train(
                    data=str(data_yaml),
                    epochs=int(hp["epochs"]),
                    imgsz=int(hp["imgsz"]),
                    batch=batch,
                    patience=int(hp["patience"]),
                    seed=int(hp["seed"]),
                    device=device,
                    workers=workers,
                    project=str(project),
                    name="train",
                    exist_ok=True,
                    plots=False,
                    verbose=False,
                    val=True,
                    deterministic=True,
                )
        finally:
            sys.stdout = real_stdout
            file_handler.flush()
        trainer = model.trainer
        if trainer is None:
            raise RuntimeError("the trainer did not run")
        best = Path(trainer.best) if Path(trainer.best).is_file() else Path(trainer.last)
        if not best.is_file():
            raise RuntimeError("the trainer produced no weights (best.pt or last.pt)")
        weights_dir = ctx.job_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        target = weights_dir / "best.pt"
        shutil.copy2(best, target)
        effective = EffectiveHyperparameters(
            epochs=int(hp["epochs"]),
            imgsz=int(hp["imgsz"]),
            batch=int(state["batch"]) if int(state["batch"]) > 0 else DEFAULT_CPU_BATCH,
            patience=int(hp["patience"]),
            seed=int(hp["seed"]),
        )
        return TrainOutcome(target, int(state["epochs_completed"]), effective, train_log, state["stop_reason"])

    def export(self, weights: Path, dest: Path, *, imgsz: int, work_dir: Path) -> ExportOutcome:
        _prepare_environment(work_dir)
        from ultralytics import YOLO  # type: ignore[attr-defined]

        model = YOLO(str(weights))
        class_names = [str(model.names[i]) for i in sorted(model.names)]
        with contextlib.redirect_stdout(sys.stderr if LOG.isEnabledFor(logging.DEBUG) else _Devnull()):
            onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False, verbose=False)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "model.onnx"
        Path(onnx_path).replace(target)
        return ExportOutcome(target, class_names, imgsz, {"kind": "yolov8", "nmsIouThreshold": NMS_IOU})


class _Devnull:
    def write(self, _s: str) -> int:
        return 0

    def flush(self) -> None:
        return None
