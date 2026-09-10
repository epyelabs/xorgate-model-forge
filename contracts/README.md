# Contracts

The forge's documents as JSON Schema, emitted from the pydantic models in
`xorgate_forge.contracts` by `make schemas` (never edited by hand; the
golden suite fails when a committed schema drifts from a fresh emission),
plus one example of each under `examples/`. The rules for what may change
between versions are in [docs/compatibility.md](../docs/compatibility.md).

| Schema | Document | Example |
| --- | --- | --- |
| `dataset.schema.json` | dataset v1 header, `dataset.json` | `examples/dataset.json` |
| `annotations.schema.json` | the canonical annotation file, `annotations.coco.json` (COCO detection, 0-based category ids, string image ids) | `examples/annotations.coco.json` |
| `splits.schema.json` | `splits.json`, image id to `train \| val \| test` | `examples/splits.json` |
| `job.schema.json` | job spec v1, `job.json` (`task: train \| autolabel`) | `examples/job-train.json`, `examples/job-autolabel.json` |
| `result.schema.json` | result v1, `result.json` | `examples/result-train.json`, `examples/result-autolabel.json` |
| `progress.schema.json` | progress v1, one line of `progress.jsonl` | `examples/progress.jsonl` |
| `labels.schema.json` | the labels file an `autolabel` job writes | `examples/autolabel-output.json` |
| `manifest-with-provenance.schema.json` | model manifest v1 as the forge writes it, with the additive `provenance` block | `examples/manifest-with-provenance.json` |
| `model-manifest.schema.json` | the engine's own manifest v1 schema, vendored from `@xorgate/workflow-engine` (`scripts/vendor-manifest-schema.mjs`) | |

The examples started as the plan's proposals (`xorgate/plans/xorgate-models/contracts/`)
and were frozen at the end of M1 with the deviations recorded in the plan's
STATUS.md; the `fixtures/` directory holds real documents the golden suite
produces and checks.

Notes a reader of the schemas needs:

- Readers ignore unknown fields, so a document written by a newer additive
  minor still parses on an older release. The engine's vendored manifest
  schema says `additionalProperties: false` because zod strips unknown keys
  on parse; the engine does not reject a manifest carrying `provenance`, it
  drops the block from the parsed object and keeps it in the file.
- `provenance` on the job spec is opaque: the forge copies it into the
  manifest (`provenance.platform`) and the result without reading it. The
  manifest's own `provenance.dataset.id` and `version` are read from the
  spec's `datasetId` and `datasetVersion` when present.
- `hyperparameters` on the job spec are overrides; the result carries the
  effective values after the preset and auto batch.
- Image ids are strings. A foreign COCO file's integer ids are accepted and
  read as their decimal strings; `xmf dataset export --to coco` writes
  sequential integer ids for other tools and keeps the string id in
  `xorgateImageId`.
