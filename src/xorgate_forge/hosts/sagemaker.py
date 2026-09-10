"""The SageMaker host: a path mapping over the local host (design 3c).

    dataset    /opt/ml/input/data/dataset          (the `dataset` channel)
    job spec   /opt/ml/input/config/job.json       (the `job` channel; the entrypoint passes it)
    models     /opt/ml/input/data/models           (an optional `models` channel: a registry root)
    output     /opt/ml/model                        (SageMaker tars it to model.tar.gz)
    progress   /opt/ml/checkpoints/progress.jsonl  (SageMaker syncs that directory to S3 continuously)
    secrets    the training job's `Environment` (XORGATE_SECRET_<NAME>)

The output directory is `/opt/ml/model` itself, without a `<jobId>` level,
because `model.tar.gz` is already per job. Everything else in the output
tree is identical to the local host's, which the golden suite proves under a
fake tree: `XMF_SAGEMAKER_ROOT` (or `--sagemaker-root`) moves `/opt/ml`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..seams import Host
from .local import LocalHost

DEFAULT_ROOT = "/opt/ml"


@dataclass(frozen=True)
class SageMakerPaths:
    root: Path
    dataset: Path
    job_spec: Path
    models: Path
    output: Path
    checkpoints: Path
    progress: Path


def sagemaker_paths(root: Path | None = None) -> SageMakerPaths:
    base = Path(root or os.environ.get("XMF_SAGEMAKER_ROOT") or DEFAULT_ROOT)
    return SageMakerPaths(
        root=base,
        dataset=base / "input" / "data" / "dataset",
        job_spec=base / "input" / "config" / "job.json",
        models=base / "input" / "data" / "models",
        output=base / "model",
        checkpoints=base / "checkpoints",
        progress=base / "checkpoints" / "progress.jsonl",
    )


class SageMakerHost:
    def __init__(self, *, job_id: str, root: Path | None = None, progress_stdout: bool = False) -> None:
        self.paths = sagemaker_paths(root)
        self.job_id = job_id
        self.progress_stdout = progress_stdout

    def bind(self) -> Host:
        p = self.paths
        models_dirs = [p.models] if p.models.is_dir() else []
        p.checkpoints.mkdir(parents=True, exist_ok=True)
        local = LocalHost(
            dataset_dir=p.dataset,
            out_dir=p.output,
            job_dir=p.output,
            job_id=self.job_id,
            models_dirs=models_dirs,
            progress_stdout=self.progress_stdout,
            extra_progress_files=[p.progress],
        )
        host = local.bind()
        host.name = "sagemaker"
        return host
