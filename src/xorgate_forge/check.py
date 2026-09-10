"""`xmf check <dir>` and `check_model_dir(dir)`: the four assertions of
`xwe models check` (engine src/local/check-model.ts) in Python, with the
same four-line output, so a job's `check` block is trustworthy without Node
in the container. The golden suite runs the real `npx xwe models check` on
the same directory and asserts the two agree.

    manifest   ok   detection/acme-forklift v1, yolov8, 2 classes, 640x640 nchw rgb
    model      ok   model.onnx 11.6 MB, sha256 verified
    session    ok   onnxruntime 1.29.0, input "images" [1,3,640,640]
    output     ok   "output0" [1,6,8400] matches yolov8 with 2 classes (4 + 2)

The `session` line names this runtime (`onnxruntime <version>`) where the
engine names its binding (`onnxruntime-node <version>`); the other three
lines are character for character the engine's. When the manifest lacks
`files`, the checksum and size are filled in unless `fill=False`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .contracts import ModelManifest
from .inference import blank_frame, dims_text, letterbox_image, ort_version
from .util import format_bytes, read_json, sha256_of

Stage = Literal["manifest", "model", "session", "output"]
STAGES: tuple[Stage, ...] = ("manifest", "model", "session", "output")
YOLO_SEG_MASK_COEFFS = 32


@dataclass
class CheckLine:
    stage: Stage
    ok: bool
    text: str

    def render(self) -> str:
        return f"{self.stage:<11}{('ok' if self.ok else 'fail'):<5}{self.text}"


@dataclass
class CheckReport:
    ok: bool
    problems: list[str]
    lines: list[CheckLine]
    manifest: ModelManifest | None = None
    manifest_updated: bool = False
    raw_manifest: dict[str, Any] | None = field(default=None, repr=False)

    def rendered(self) -> list[str]:
        return [line.render() for line in self.lines]


def _first_issue(err: ValidationError) -> str:
    issue = err.errors()[0]
    loc = ".".join(str(p) for p in issue.get("loc", ())) or "(root)"
    return f"{loc}: {issue.get('msg', 'invalid')}"


def _check_output_shape(manifest: ModelManifest, outputs: list[tuple[str, list[int]]]) -> tuple[bool, str]:
    kind = manifest.postprocess.kind
    classes = len(manifest.classNames)
    shapes = ", ".join(f'"{n}" {dims_text(d)}' for n, d in outputs)
    if kind == "yolov8":
        head = next(((n, d) for n, d in outputs if len(d) == 3), None)
        if head is None:
            return False, f"expected one rank-3 detection head for yolov8, got {shapes}"
        name, dims = head
        rows = dims[1]
        expected = 4 + classes
        if dims[0] != 1 or rows != expected:
            implied = rows - 4
            hint = f" (a model with {implied} classes?)" if implied > 0 else ""
            return (
                False,
                f'"{name}" {dims_text(dims)} does not match yolov8 with {classes} classes: '
                f"expected 4 + {classes} = {expected} rows, got {rows}{hint}",
            )
        return True, f'"{name}" {dims_text(dims)} matches yolov8 with {classes} classes (4 + {classes})'
    if kind == "yolov8-seg":
        head = next(((n, d) for n, d in outputs if len(d) == 3), None)
        proto = next(((n, d) for n, d in outputs if len(d) == 4), None)
        if head is None or proto is None:
            return False, f"expected a rank-3 detection head and a rank-4 prototype tensor for yolov8-seg, got {shapes}"
        rows = head[1][1]
        expected = 4 + classes + YOLO_SEG_MASK_COEFFS
        if rows != expected or proto[1][1] != YOLO_SEG_MASK_COEFFS:
            return (
                False,
                f'"{head[0]}" {dims_text(head[1])} + "{proto[0]}" {dims_text(proto[1])} do not match yolov8-seg with '
                f"{classes} classes: expected 4 + {classes} + {YOLO_SEG_MASK_COEFFS} = {expected} rows and "
                f"{YOLO_SEG_MASK_COEFFS} prototypes, got {rows} rows and {proto[1][1]} prototypes",
            )
        return (
            True,
            f'"{head[0]}" {dims_text(head[1])} + "{proto[0]}" {dims_text(proto[1])} match yolov8-seg with '
            f"{classes} classes (4 + {classes} + {YOLO_SEG_MASK_COEFFS})",
        )
    if kind == "depth-anything":
        name, dims = outputs[0]
        leading, mh, mw = dims[:-2], dims[-2], dims[-1]
        if len(dims) < 3 or any(d != 1 for d in leading) or not (mh > 0 and mw > 0):
            return False, f'"{name}" {dims_text(dims)} does not match depth-anything: expected one map [1,H,W]'
        return True, f'"{name}" {dims_text(dims)} matches depth-anything (one {mw}x{mh} map)'
    return False, f'postprocess.kind "{kind}" has no decoder in this engine ({shapes})'


def check_model_dir(dir_path: Path, *, fill: bool = True) -> CheckReport:
    d = Path(dir_path)
    lines: list[CheckLine] = []
    problems: list[str] = []
    manifest: ModelManifest | None = None
    manifest_updated = False
    raw: dict[str, Any] | None = None

    def push(stage: Stage, ok: bool, text: str) -> None:
        lines.append(CheckLine(stage, ok, text))
        if not ok:
            problems.append(f"{stage}: {text}")

    def skip_rest(from_stage: Stage, reason: str) -> CheckReport:
        for stage in STAGES[STAGES.index(from_stage) :]:
            lines.append(CheckLine(stage, False, reason))
        return CheckReport(False, problems, lines, manifest, manifest_updated, raw)

    manifest_path = d / "manifest.json"
    if not d.is_dir():
        push("manifest", False, f"{d} is not a directory")
        return skip_rest("model", "skipped")
    if not manifest_path.is_file():
        push("manifest", False, f"manifest.json not found in {d}")
        return skip_rest("model", "skipped")
    try:
        raw = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as err:
        push("manifest", False, f"manifest.json is not JSON: {err}")
        return skip_rest("model", "skipped")
    try:
        manifest = ModelManifest.model_validate(raw)
    except ValidationError as err:
        push("manifest", False, f"manifest.json: {_first_issue(err)}")
        return skip_rest("model", "skipped")
    n = len(manifest.classNames)
    classes_text = "no class names" if n == 0 else f"{n} classes"
    push(
        "manifest",
        True,
        f"{manifest.family}/{manifest.name} v{manifest.version}, {manifest.postprocess.kind}, {classes_text}, "
        f"{manifest.input.width}x{manifest.input.height} {manifest.input.layout} {manifest.input.colorSpace}",
    )

    model_path = d / "model.onnx"
    if not model_path.is_file():
        push("model", False, f"model.onnx not found in {d}")
        return skip_rest("session", "skipped")
    size = model_path.stat().st_size
    digest = sha256_of(model_path)
    declared = (manifest.files or {}).get("model.onnx")
    if declared is not None:
        if declared.sha256 != digest:
            push(
                "model",
                False,
                f"model.onnx {format_bytes(size)}, sha256 mismatch "
                f"(manifest {declared.sha256[:12]}…, file {digest[:12]}…)",
            )
            return skip_rest("session", "skipped")
        if declared.sizeBytes != size:
            push("model", False, f"model.onnx is {size} bytes, the manifest declares {declared.sizeBytes}")
            return skip_rest("session", "skipped")
        push("model", True, f"model.onnx {format_bytes(size)}, sha256 verified")
    elif fill:
        assert raw is not None
        files = dict(raw.get("files") or {})
        files["model.onnx"] = {"sha256": digest, "sizeBytes": size}
        updated = {**raw, "files": files}
        manifest_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        manifest = ModelManifest.model_validate(updated)
        raw = updated
        manifest_updated = True
        push("model", True, f"model.onnx {format_bytes(size)}, sha256 recorded in manifest.json")
    else:
        push("model", True, f"model.onnx {format_bytes(size)}, no sha256 in manifest (not verified)")

    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception as err:
        push("session", False, str(err))
        return skip_rest("output", "skipped")
    inp = session.get_inputs()[0]
    input_shape = (
        [dim if isinstance(dim, int) else str(dim) for dim in inp.shape]
        if inp.shape
        else [
            1,
            3,
            manifest.input.height,
            manifest.input.width,
        ]
    )
    push("session", True, f'onnxruntime {ort_version()}, input "{inp.name}" {dims_text(input_shape)}')

    try:
        lb = letterbox_image(blank_frame(640, 480), manifest.input)
        results = session.run(None, {inp.name: lb.tensor})
        outputs = [(o.name, [int(x) for x in r.shape]) for o, r in zip(session.get_outputs(), results)]
        if not outputs:
            push("output", False, "the session produced no outputs")
        else:
            ok, text = _check_output_shape(manifest, outputs)
            push("output", ok, text)
    except Exception as err:
        push("output", False, f"inference failed: {err}")

    return CheckReport(len(problems) == 0, problems, lines, manifest, manifest_updated, raw)
