"""The local host: directories on disk, `progress.jsonl` in the job's output
directory and, with `--progress json`, NDJSON on stdout; secrets from
`XORGATE_SECRET_<NAME>` or a `--secrets file.json`.

Model resolution follows the engine's local host: explicit `--models-dir`
roots first, then the cache `~/.xorgate/models` (`XWE_MODELS_DIR`), the same
directory `xwe models pull` fills. Base weights (`yolov8n.pt`) are looked for
under `XMF_WEIGHTS_DIR` (default `/opt/forge/weights`, where the images bake
them), then the forge's own cache `~/.xorgate/forge/weights`, and are
downloaded there only when `XMF_WEIGHTS_PULL` is not `0`."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import IO, Any

from ..contracts import JobResult, ModelManifest, ProgressMessage
from ..seams import Host, ModelNotFoundError, ResolvedModel, WeightsNotFoundError
from ..util import LOG, read_json, write_json_atomic

SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_WEIGHTS_DIR = "/opt/forge/weights"


def default_cache_dir() -> Path:
    env = os.environ.get("XWE_MODELS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".xorgate" / "models"


def default_weights_cache_dir() -> Path:
    return Path.home() / ".xorgate" / "forge" / "weights"


def baked_weights_dir() -> Path:
    return Path(os.environ.get("XMF_WEIGHTS_DIR") or DEFAULT_WEIGHTS_DIR).expanduser()


def parse_model_name(family: str, raw: str) -> tuple[str, str, int | None]:
    """`name[@version]` as node config carries it (engine src/local/model-ref.ts)."""
    raw = raw.strip()
    version: int | None = None
    base = raw
    at = raw.rfind("@")
    if at != -1:
        base, tail = raw[:at], raw[at + 1 :]
        if not tail.isdigit() or int(tail) < 1:
            raise ValueError(f'model "{raw}": the version after "@" must be a positive integer')
        version = int(tail)
    if not SEGMENT_RE.match(base):
        raise ValueError(f'model "{raw}": not a valid model name')
    if not SEGMENT_RE.match(family):
        raise ValueError(f'model family "{family}" is not valid')
    return family, base, version


def find_model_in_root(root: Path, family: str, name: str, version: int | None) -> ResolvedModel | None:
    """Highest complete version under `<root>/<family>/<name>/`, or the pinned one."""
    name_dir = root / family / name
    if not name_dir.is_dir():
        return None
    versions = sorted(
        (int(p.name) for p in name_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        reverse=True,
    )
    for v in versions:
        if version is not None and v != version:
            continue
        d = name_dir / str(v)
        manifest_path = d / "manifest.json"
        model_path = d / "model.onnx"
        if not manifest_path.is_file() or not model_path.is_file():
            continue
        manifest = ModelManifest.model_validate(read_json(manifest_path))
        return ResolvedModel(family, name, v, manifest, d, model_path, root)
    return None


class DirDatasetSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def resolve(self) -> Path:
        return self.path


class LocalModelSource:
    def __init__(
        self,
        roots: list[Path] | None = None,
        cache_dir: Path | None = None,
        weights_dirs: list[Path] | None = None,
        allow_download: bool | None = None,
    ) -> None:
        self.roots = [Path(r) for r in (roots or [])]
        self.cache_dir = cache_dir or default_cache_dir()
        self.weights_dirs = weights_dirs or [baked_weights_dir(), default_weights_cache_dir()]
        if allow_download is None:
            allow_download = os.environ.get("XMF_WEIGHTS_PULL", "1") != "0"
        self.allow_download = allow_download

    def search_roots(self) -> list[Path]:
        return [*self.roots, self.cache_dir]

    def fetch(self, family: str, name: str) -> ResolvedModel:
        family, base, version = parse_model_name(family, name)
        for root in self.search_roots():
            found = find_model_in_root(root, family, base, version)
            if found is not None:
                LOG.info(f"model {family}/{base} v{found.version} <- {found.model_path}")
                return found
        pin = f"@{version}" if version is not None else ""
        roots = ", ".join(str(r) for r in self.search_roots())
        raise ModelNotFoundError(
            f"model {family}/{base}{pin} not found under {roots} (run `xwe models pull {family}/{base}{pin}`, "
            f"or pass --models-dir)"
        )

    def fetch_weights(self, ref: str) -> Path:
        if not SEGMENT_RE.match(ref):
            raise ValueError(f'base weights "{ref}": not a valid reference')
        filename = ref if ref.endswith(".pt") else f"{ref}.pt"
        for d in self.weights_dirs:
            candidate = d / filename
            if candidate.is_file():
                LOG.info(f"weights {ref} <- {candidate}")
                return candidate
        if not self.allow_download:
            looked = ", ".join(str(d) for d in self.weights_dirs)
            raise WeightsNotFoundError(
                f"weights {ref} not found under {looked} and downloads are off (XMF_WEIGHTS_PULL=0)"
            )
        target_dir = self.weights_dirs[-1]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        LOG.info(f"weights {ref}: downloading to {target}")
        from ..recipes.yolov8 import download_ultralytics_asset  # heavy import, keep local

        download_ultralytics_asset(target)
        if not target.is_file():
            raise WeightsNotFoundError(f"weights {ref}: download did not produce {target}")
        return target

    def describe(self) -> list[str]:
        lines = [f"models: {r}" for r in self.search_roots()]
        lines += [f"weights: {d}" for d in self.weights_dirs]
        return lines


class LocalArtifactSink:
    def __init__(self, job_dir: Path) -> None:
        self.job_dir = Path(job_dir)

    def dir(self) -> Path:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        return self.job_dir

    def finalize(self, result: JobResult) -> Path:
        path = self.dir() / "result.json"
        write_json_atomic(path, result.to_json_dict())
        return path


class FileProgressSink:
    """Appends every message to one or more `progress.jsonl` files and,
    optionally, to a stream (stdout for `--progress json`). The stream
    handle is captured at construction, so a trainer that later hijacks
    `sys.stdout` cannot get between the sink and the real stdout."""

    def __init__(self, files: list[Path], stream: IO[str] | None = None) -> None:
        self.files = [Path(f) for f in files]
        self._handles: list[IO[str]] = []
        for f in self.files:
            f.parent.mkdir(parents=True, exist_ok=True)
            self._handles.append(f.open("a", encoding="utf-8"))
        self.stream = stream

    def emit(self, message: ProgressMessage) -> None:
        line = json.dumps(message.to_json_dict(), separators=(",", ":"))
        for h in self._handles:
            h.write(line + "\n")
            h.flush()
        if self.stream is not None:
            self.stream.write(line + "\n")
            self.stream.flush()

    def close(self) -> None:
        for h in self._handles:
            h.close()
        self._handles = []


class EnvSecretReader:
    """`XORGATE_SECRET_<LOGICAL_NAME>` (upper-cased, `-` and `.` to `_`), or a
    JSON file of `{ "<logical-name>": "<value>" }`. Values are never logged."""

    def __init__(self, file: Path | None = None, environ: dict[str, str] | None = None) -> None:
        self.environ = dict(os.environ if environ is None else environ)
        self.file_values: dict[str, str] = {}
        if file is not None:
            data: Any = read_json(Path(file))
            if not isinstance(data, dict):
                raise ValueError(f"{file}: expected a JSON object of logical names to values")
            self.file_values = {str(k): str(v) for k, v in data.items()}

    @staticmethod
    def env_name(logical_name: str) -> str:
        return "XORGATE_SECRET_" + re.sub(r"[^A-Za-z0-9]", "_", logical_name).upper()

    def get(self, logical_name: str) -> str | None:
        if logical_name in self.file_values:
            return self.file_values[logical_name]
        return self.environ.get(self.env_name(logical_name))


class LocalHost:
    """`xmf run job.json --dataset ./ds --out ./out [--models-dir d]...`."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        out_dir: Path,
        job_id: str,
        models_dirs: list[Path] | None = None,
        progress_stdout: bool = False,
        secrets_file: Path | None = None,
        extra_progress_files: list[Path] | None = None,
        job_dir: Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.out_dir = Path(out_dir)
        self.job_id = job_id
        self.models_dirs = [Path(d) for d in (models_dirs or [])]
        self.progress_stdout = progress_stdout
        self.secrets_file = secrets_file
        self.extra_progress_files = extra_progress_files or []
        self.job_dir = job_dir or (self.out_dir / job_id)

    def bind(self) -> Host:
        sink = LocalArtifactSink(self.job_dir)
        job_dir = sink.dir()
        progress_files = [job_dir / "progress.jsonl", *self.extra_progress_files]
        return Host(
            name="local",
            dataset=DirDatasetSource(self.dataset_dir),
            models=LocalModelSource(roots=self.models_dirs),
            artifacts=sink,
            progress=FileProgressSink(progress_files, stream=sys.stdout if self.progress_stdout else None),
            secrets=EnvSecretReader(self.secrets_file),
        )
