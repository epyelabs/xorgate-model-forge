# Changelog

All notable changes to `xorgate-forge` (`xmf`). The version rules are in
[docs/compatibility.md](./docs/compatibility.md).

## Unreleased

## 0.2.0 - 2026-09-14

The first tagged release. Everything listed under 0.1.0 below shipped as the
untagged images `ghcr.io/epyelabs/xmf:0.1.0`; this release adds one field to
two contracts and is otherwise the same package.

### Added

- `metrics.confidence` on result v1 and `provenance.metrics.confidence` on
  the model manifest: the operating confidence, the threshold that
  maximizes F1 on the validation split at IoU 0.5, at which `precision`
  and `recall` are reported (the value the job log already printed as "at
  confidence 0.05"). Additive and optional; a reader of 0.1.0 documents
  sees no value. A detection node running the model should use it as its
  minimum confidence: a small fine-tune can rank its classes well (a high
  mAP50) while every score sits under a generic default such as 0.1, and
  the platform's picker now prefills the node's threshold from this field.
  The value is null when no threshold on the grid yields a true positive
  (precision and recall then read 0): such a model has no operating point
  and nothing should prefill from it. Schemas and examples regenerated; the
  golden suite asserts the field in both documents and that the two agree.

## 0.1.0 - 2026-09-11

### Added

- The four forge contracts (dataset v1, job spec v1, result v1, progress v1),
  the labels file, and the engine's model manifest v1 with the additive
  `provenance` block, as pydantic models with emitted JSON Schemas under
  `contracts/`.
- Five host seams, `LocalHost`, and `SageMakerHost` as a path mapping over
  it (`/opt/ml`, relocatable with `XMF_SAGEMAKER_ROOT`).
- The `yolov8` recipe (Ultralytics, AGPL-3.0): presets `fast` and
  `accurate`, per-epoch progress, early stop by `patience`, a clean stop at
  `limits.maxMinutes`, ONNX export at opset 12 and a static `imgsz` with
  manifest v1 and `files`, evaluation through the exported ONNX (mAP50,
  mAP50-95, precision, recall, per class, confusion matrix, sample
  renders), and `autolabel` with any registry model.
- `xmf doctor | recipes | validate | run | train | check | weights pull` and
  `xmf dataset validate | import | export | split | stats`; exit codes
  0/1/2; `--progress json`; `--log-level`.
- The fixture dataset (63 frames of the engine's `drive.mp4`, pre-labeled
  with `yolov8n`), the golden suite (`make test`), a Dockerfile with `cpu`
  and `cuda` targets that bake the base weights, `ci.yml` and a manual
  `release.yml`.
