# xorgate-model-forge

The xorgate model forge as one Python package and one container image. It
trains, evaluates, auto-labels, exports and checks object detection models
against the workflow engine's frozen model contract (manifest v1 and the
registry layout), with no platform account and no network once the base
weights are on disk. The same package is the `xmf` command, a library you
embed, and a host contract the platform implements over S3 and SageMaker.

Documentation: <https://docs.xorgate.io/docs/models> (the Models section is
the manual; the forge CLI page describes this package).
Compatibility policy: [docs/compatibility.md](./docs/compatibility.md).
Contracts: [contracts/](./contracts/). Changes: [CHANGELOG.md](./CHANGELOG.md).
License: MIT. The trainers a recipe wraps carry their own licenses; `xmf
recipes` prints them (`yolov8` is Ultralytics, AGPL-3.0).

## Install

```bash
pip install xorgate-forge                 # contracts, dataset tools, check (pydantic, numpy, pillow, onnx)
pip install "xorgate-forge[yolov8]"       # the yolov8 recipe: ultralytics, torch, onnxruntime
pip install "xorgate-forge[assist]"       # auto-labeling and evaluation with any registry model (onnxruntime)
pip install "xorgate-forge[all]"
```

Python 3.11 or newer. On Linux, install torch from the CPU wheel index
first when you do not want the CUDA stack (`pip install torch torchvision
--index-url https://download.pytorch.org/whl/cpu`); the checkout's `uv.lock`
already does that. `xmf doctor` tells you what a machine is missing and how
to fix it.

The image bundles Python, torch, ONNX Runtime and the base weights of every
recipe, so a job runs with network isolation on:

```bash
docker run --rm -v "$PWD":/work ghcr.io/epyelabs/xmf doctor
docker run --rm -v "$PWD":/work ghcr.io/epyelabs/xmf run job.json --dataset ds --out out
docker run --rm --gpus all -v "$PWD":/work ghcr.io/epyelabs/xmf:0.1.0-cuda run job.json --dataset ds --out out
```

`ghcr.io/epyelabs/xmf:<version>` is the CPU image (linux/amd64 and
linux/arm64); `:<version>-cuda` is the CUDA image (linux/amd64) for GPU
training jobs. Without arguments the image runs `xmf run
/opt/ml/input/config/job.json --host sagemaker`, the SageMaker training-job
shape.

## Run

```bash
xmf doctor                                                   # python, torch, CUDA, ONNX Runtime, recipes, weights, cache
xmf recipes                                                  # id, family, task, license, base weights
xmf validate job.json --dataset ./ds                         # what a job needs; exit 2 on a bad spec
xmf run job.json --dataset ./ds --out ./out                  # artifacts land in out/<jobId>/
xmf run job.json --dataset ./ds --out ./out --progress json  # NDJSON progress on stdout
xmf train job.json --dataset ./ds --out ./out                # run, task train only
xmf dataset validate ./ds                                    # every image opens, every box is in bounds, every class has val instances
xmf dataset import ./coco.json --from coco --out ./ds        # also --from yolo | voc
xmf dataset export ./ds --to yolo --out ./yolo               # also --to coco
xmf dataset split ./ds --train 0.8 --val 0.2 --seed 0        # by group, deterministic; --keep <previous splits.json>
xmf dataset stats ./ds                                       # counts per split and class, box sizes, unlabeled images
xmf check ./out/<jobId>/model/detection/forklift/1           # the four-line check, same as `xwe models check`
xmf weights pull yolov8n yolov8s                             # base weights into ~/.xorgate/forge/weights
```

- **A job** is one JSON document (`job.json`, job spec v1): `task: train`
  names a recipe, the base weights, a `mode` (`fast` or `accurate`, a
  recipe preset the `hyperparameters` override), `limits.maxMinutes`, and
  the registry identity of the output (`output.family`, `name`, `version`,
  `label`); `task: autolabel` names a registry model, `minConfidence` and a
  `classMap` from the model's classes to the dataset's.
  [contracts/examples/](./contracts/examples/) has one of each.
- **A dataset** is a directory (dataset v1): `dataset.json`, one
  `annotations.coco.json` (COCO detection, category ids are 0-based indexes
  into `classes`), `splits.json`, and `images/`. `xmf dataset import` builds
  one from COCO, YOLO or Pascal VOC; `xmf dataset validate` is the check a
  job runs before training.
- **Artifacts**: `out/<jobId>/result.json` (written last, always, even on
  failure), `progress.jsonl`, `model/<family>/<name>/<version>/{manifest.json,
  model.onnx}` (the registry layout, so `xwe run --models-dir
  out/<jobId>/model` loads it unchanged), `weights/best.pt`,
  `eval/{metrics.json, confusion.json, samples/<n>.jpg}` and `train.log`.
- **The manifest** is model manifest v1 exactly as the engine's
  `tools/export/export_yolo.py` writes it, plus an additive `provenance`
  block (the forge version, the recipe and its license, the base, the job
  id, the dataset hash, the metrics, the training time, and the opaque
  `provenance` block of the job spec). Engine 0.3.x keeps it in the file
  and drops it on parse.
- **Check and evaluate** run on the exported ONNX, not the PyTorch weights:
  the check is the same four assertions as `xwe models check` with the same
  four-line output (the `session` line names this runtime); evaluate runs
  the manifest's own letterbox and decode over the validation split, so
  the metrics describe the artifact the engine will run.
- **Models** for `autolabel` resolve from `--models-dir` roots, then the
  cache (`XWE_MODELS_DIR`, default `~/.xorgate/models`, the same directory
  `xwe models pull` fills). Base weights resolve from `XMF_WEIGHTS_DIR`
  (default `/opt/forge/weights`, where the images bake them), then
  `~/.xorgate/forge/weights`, and are downloaded there unless
  `XMF_WEIGHTS_PULL=0`.
- **Secrets** come from the environment (`XORGATE_SECRET_<LOGICAL_NAME>`)
  or a `--secrets file.json`; a value never appears in an artifact or an
  output line. No Phase 1 recipe needs one.
- **Progress**: one object per line, `t: status | epoch | log`, each
  stamped `at`; `status` carries `phase: prepare | train | export | check |
  evaluate | label | done`, `epoch` carries `epoch`, `epochs`, `loss` and
  `mAP50`. The same objects go to `progress.jsonl` and, with `--progress
  json`, to stdout.
- **Limits**: training stops cleanly at `limits.maxMinutes` minus a reserve
  for export, check and evaluate (20 percent, at most five minutes), then
  the job finishes on the best weights so far.
- **Exit codes**: `0` success; `1` the job failed (`result.json` written
  with `error.phase` and `error.message`) or a validation found problems;
  `2` usage or configuration, nothing ran.
- **Logging**: `--log-level silent | error | info | debug` on any command
  gates stderr only; stdout carries results and the progress stream. The
  trainer's own output goes to `train.log`.

## Embed

```python
from pathlib import Path
from xorgate_forge.hosts import LocalHost
from xorgate_forge.runner import load_spec, run_job

spec = load_spec(Path("job.json"))
host = LocalHost(dataset_dir=Path("ds"), out_dir=Path("out"), job_id=spec.jobId).bind()
outcome = run_job(spec, host)
print(outcome.result.status, outcome.result.metrics, outcome.job_dir)
```

`run_job` resolves to a `RunOutcome` with the parsed `result` and the job
directory; a failed job comes back with `result.status == "failed"` and
`result.json` written. The contracts are pydantic models under
`xorgate_forge.contracts`; `check_model_dir` (`xorgate_forge.check`),
`evaluate_model` (`xorgate_forge.evaluate`), the dataset tools
(`xorgate_forge.dataset`) and the ONNX runner (`xorgate_forge.inference`)
are importable on their own. Importing the package loads no trainer; torch
and Ultralytics are imported by the recipe when a job runs.

## Hosts

The runner reads and writes only through five small seams
(`xorgate_forge.seams`): a dataset source, a model source, an artifact
sink, a progress sink and a secret reader. `LocalHost` is the reference
implementation and short enough to read; `SageMakerHost` is a path mapping
over it:

| Seam | Local host | SageMaker host |
| --- | --- | --- |
| dataset | `--dataset <dir>` | `/opt/ml/input/data/dataset` |
| job spec | the `xmf run` argument | `/opt/ml/input/config/job.json` |
| models | `--models-dir` roots, then `~/.xorgate/models`; weights under `XMF_WEIGHTS_DIR` | `/opt/ml/input/data/models` (optional channel), then baked weights |
| artifacts | `<out>/<jobId>/`, `result.json` last | `/opt/ml/model/` (SageMaker tars it to `model.tar.gz`) |
| progress | `progress.jsonl` in the job dir, NDJSON on stdout with `--progress json` | `/opt/ml/checkpoints/progress.jsonl` (synced to S3 while the job runs) |
| secrets | `XORGATE_SECRET_<NAME>`, `--secrets file.json` | the job's `Environment` |

`xmf run job.json --host sagemaker` selects the mapping; `XMF_SAGEMAKER_ROOT`
(or `--sagemaker-root`) moves `/opt/ml`, which is how the golden suite proves
the host under a fake tree without SageMaker. The output tree under
`/opt/ml/model` is byte-for-byte the local host's tree under
`<out>/<jobId>`.

## Developing

```bash
make setup               # uv sync (all extras + dev) and npm ci (the engine, for the two `npx xwe` cases)
make lint                # ruff check + format --check
make typecheck           # mypy --strict over src/
make test                # pulls the fixtures, then the golden suite; prints the suite duration
make schemas             # re-emit contracts/*.schema.json from the pydantic models and the engine
make fixture-dataset     # rebuild fixtures/ds from drive.mp4 with the pulled yolov8n
```

`make test` first runs `scripts/fetch_fixtures.py`, which fetches the two
large golden fixtures (the engine's 25 s `drive.mp4` and the `yolov8n`
model) per `fixtures/fixtures.json` from the public registry, or copies
them from the engine checkout next to this repo when it has them;
`XMF_FIXTURES_URL` points it at a mirror. `fixtures/ds` is the committed
fixture dataset: 63 frames of `drive.mp4` pre-labeled with `yolov8n` for
`car` and `truck`, hand-checked once. The suite trains three times (the
quick spec, a one-epoch run through the SageMaker host, and a one-minute
budget run), so it takes minutes on a laptop CPU. Node is a dev dependency
only for the cases that run the real `npx xwe models check` and `npx xwe
run` on what the forge produced; `XMF_GOLDEN_SKIP_TRAIN=1` runs the
contract, dataset and CLI cases alone.

Images:

```bash
make image-cpu             # ghcr.io/epyelabs/xmf:dev for the native architecture
make image-cpu-multiarch   # linux/amd64 + linux/arm64 with buildx
make image-cuda            # ghcr.io/epyelabs/xmf:<version>-cuda, linux/amd64
docker run --rm -v "$PWD":/work ghcr.io/epyelabs/xmf:dev run fixtures/specs/quick.json --dataset fixtures/ds --out out-docker
```

Both targets bake the recipes' base weights under `/opt/forge/weights` at
build time (the only network access a training job needs), set `WORKDIR
/work` for the mounted local case, and default to the SageMaker entrypoint.

CI (`.github/workflows/ci.yml`) runs lint, typecheck and the golden suite on
Linux x64 and macOS, and builds the cpu image. `release.yml` (manual
trigger) builds and pushes both images to `ghcr.io/epyelabs/xmf`; the
version in `pyproject.toml` must match the `v*` tag it is run for.

## Provenance

The forge is the training-side counterpart of `@xorgate/workflow-engine`
(the engine is a read-only dependency here: its model manifest v1, registry
layout and `fetchModel` seam are the hand-off contract, frozen). The
`yolov8` export folds in the engine's `tools/export/export_yolo.py`; the
check re-implements `src/local/check-model.ts`; the letterbox and YOLOv8
decode mirror `src/engine/vision.ts`. The plan that produced it is
`xorgate/plans/xorgate-models/` (design §3 is the forge).
