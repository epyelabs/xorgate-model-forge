# Compatibility

What `xorgate-forge` promises across versions, and how its version number
moves. The forge is four documents, a labels file, a manifest, five host
seams, a progress stream, a CLI and its exit codes, with code behind them;
the contracts are what the platform, a host implementer and an operator
depend on, so they are what the version number is about.

## The contracts

| Contract | Identity | Defined in | Schema | Who depends on it |
| --- | --- | --- | --- | --- |
| Dataset | `datasetVersion: 1`: `dataset.json`, `annotations.coco.json` (COCO detection, 0-based category ids, string image ids), `splits.json`, `images/` | `xorgate_forge.contracts.dataset` | `contracts/dataset.schema.json`, `annotations.schema.json`, `splits.schema.json` | the platform's version builder, `xmf dataset *`, every recipe |
| Job spec | `specVersion: 1`, `task: train \| autolabel`, discriminated on `task` | `xorgate_forge.contracts.job` | `contracts/job.schema.json` | the platform's dispatcher, operators |
| Result | `resultVersion: 1`, written last, always; `status: succeeded \| failed`, `error {phase, message}`; `metrics.confidence` (the operating confidence) since 0.2.0 | `xorgate_forge.contracts.result` | `contracts/result.schema.json` | the platform's finalizer, tooling |
| Progress | one object per line, `t: status \| epoch \| log`, each stamped `at`; `phase: prepare \| train \| export \| check \| evaluate \| label \| done` | `xorgate_forge.contracts.progress` | `contracts/progress.schema.json` | the platform's job endpoint, `--progress json` consumers |
| Labels | `labelsVersion: 1`, the `autolabel` output | `xorgate_forge.contracts.labels` | `contracts/labels.schema.json` | the platform's annotation import |
| Model manifest | the engine's manifest v1 (vendored schema, `contracts/model-manifest.schema.json`) plus the additive optional `provenance` block; `provenance.metrics.confidence` since 0.2.0 | `xorgate_forge.contracts.manifest` | `contracts/manifest-with-provenance.schema.json` | registries, `xmf check`, `xwe models check`, engine 0.4.0's typed `provenance` |
| Output tree | `<job dir>/result.json`, `progress.jsonl`, `model/<family>/<name>/<version>/{manifest.json, model.onnx}`, `weights/best.pt`, `eval/{metrics.json, confusion.json, samples/<n>.jpg}`, `train.log`, `labels.json` | `xorgate_forge.runner` | | hosts, the platform's finalizer, `xwe run --models-dir` |
| Host seams | `DatasetSource`, `ModelSource`, `ArtifactSink`, `ProgressSink`, `SecretReader`; `Host` | `xorgate_forge.seams` | | host implementers (`LocalHost`, `SageMakerHost`, the platform) |
| SageMaker mapping | `/opt/ml/input/data/dataset`, `/opt/ml/input/config/job.json`, `/opt/ml/input/data/models`, `/opt/ml/model`, `/opt/ml/checkpoints/progress.jsonl`; `XMF_SAGEMAKER_ROOT` | `xorgate_forge.hosts.sagemaker` | | the platform's training-job request builder |
| Check lines | four stages `manifest`, `model`, `session`, `output`, rendered `<stage padded to 11><ok\|fail padded to 5><text>`, the engine's wording | `xorgate_forge.check` | | `result.check.lines`, operators |
| Recipes | `id`, `family`, `task`, `license`, `presets {fast, accurate}`, `base_refs` | `xorgate_forge.recipes` | | `xmf recipes`, the platform's train dialog |
| CLI exit codes | `0` success, `1` the job or the validation failed (`result.json` written for runs), `2` usage or configuration (nothing ran) | `xorgate_forge.cli.main` | | scripts, CI, the platform's finalizer |
| CLI commands and flags | `doctor`, `recipes`, `validate`, `run`, `train`, `dataset validate \| import \| export \| split \| stats`, `check`, `weights pull`; `--log-level`, `--progress json`, `--host`, `--models-dir`, `--secrets` | `xorgate_forge.cli.main` | | operators |
| Environment | `XWE_MODELS_DIR`, `XMF_WEIGHTS_DIR`, `XMF_WEIGHTS_PULL`, `XMF_SAGEMAKER_ROOT`, `XMF_DEVICE`, `XMF_WORKERS`, `XORGATE_SECRET_<LOGICAL_NAME>`, `XMF_FIXTURES_URL` | `xorgate_forge.hosts.local`, `recipes/yolov8.py`, `scripts/fetch_fixtures.py` | | operators |
| Python | `>=3.11` | `pyproject.toml` | | everyone |

Everything else is implementation and may change in any release: the
wording of log lines and of `progress` `log` messages, the exact metrics a
training run reaches on the same inputs (they move with torch, Ultralytics
and the hardware), the bytes of `model.onnx` and of the sample renders, the
temp layout the recipe materializes, and the fields of `eval/metrics.json`
beyond the ones `result.metrics` names.

## The version number

The package follows semantic versioning, with the contracts above as the
subject:

- **Major** (`x.0.0`): a breaking change to any contract in the table. A
  field removed or retyped, a seam method renamed or given a new required
  parameter, a progress message type or a phase removed, an exit code
  reassigned, a CLI command removed, the Python floor raised past a
  version that was supported.
- **Minor** (`0.x.0`): an additive change. A new optional field in a
  document, a new optional seam method, a new message type or phase, a
  new command or flag, a new recipe, a new environment variable.
- **Patch** (`0.0.x`): no contract changes. Bug fixes, dependency updates,
  documentation. A fix that changes what a recipe produces on the same
  input is a patch, and the changelog names the recipe under a "Recipe
  behaviour" heading.

Two rules the table implies:

- **Schema versions move only on majors.** `datasetVersion`,
  `specVersion`, `resultVersion` and `labelsVersion` stay at 1 while the
  package stays at 0.x and 1.x additive; a document written by an older
  release keeps parsing on every newer release of the same major, and a
  document written by a newer minor may carry fields an older release
  ignores (readers drop unknown fields).
- **The schemas in `contracts/` are emitted, not written.** `make
  schemas` regenerates them from the pydantic models (and the manifest
  schema from the engine package); the golden suite fails when the
  committed copies drift.

Additive changes so far, by the release that introduced them:

| Release | Contract | Change |
| --- | --- | --- |
| 0.2.0 | Result, Model manifest | `metrics.confidence` and `provenance.metrics.confidence`: the operating confidence (the F1-maximizing threshold on the validation split at IoU 0.5) at which `precision` and `recall` are reported; null when no threshold yields a true positive. Optional; a 0.1.0 document has no value and every 0.1.0 reader ignores it. |

While the package is `0.x`, semantic versioning would allow breaking
changes on minors. This package does not use that latitude: the rules
above apply from `0.1.0`, and the four forge contracts were frozen at the
end of the plan's M1. The intent is that `1.0.0` changes nothing but the
number.

## The engine

The forge tracks `@xorgate/workflow-engine` on one axis only: the model
manifest v1 and the registry layout. `contracts/model-manifest.schema.json`
is vendored from the engine version in `package.json`; a new engine minor
that names `provenance` changes nothing here (the block is already written
in the shape the distribution design reserved).

## Releasing

Releases are tags. The release workflow (`.github/workflows/release.yml`)
is triggered by hand on a `v*` tag, refuses a tag that does not equal
`v<pyproject version>`, runs the gate (lint, typecheck, the golden suite),
then builds and pushes `ghcr.io/epyelabs/xmf:<version>` (linux/amd64 and
linux/arm64, the cpu target) and `ghcr.io/epyelabs/xmf:<version>-cuda`
(linux/amd64), plus `:latest` and `:latest-cuda` for a stable version. The
repository needs the `packages: write` permission the workflow declares
and the GHCR package linked to it. PyPI publishing is a later step (plan
Q9); the workflow has the job stubbed and disabled.

The changelog (`CHANGELOG.md`) has one section per version, with the
contract changes and any recipe behaviour changes named explicitly.
