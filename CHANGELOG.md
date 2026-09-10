# Changelog

All notable changes to `xorgate-forge` (`xmf`). The version rules are in
[docs/compatibility.md](./docs/compatibility.md).

## Unreleased

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
