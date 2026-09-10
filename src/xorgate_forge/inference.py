"""Running a registry model the way the engine does: the manifest's own
letterbox and normalization in, the `postprocess.kind` decoder out
(engine src/engine/vision.ts `letterboxToTensor`, `postprocessYoloV8`,
`nmsPerClass`). Evaluate and assist both go through here, so the metrics
describe the ONNX artifact the engine will run, not the PyTorch weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .contracts import ManifestInput, ModelManifest

MAX_DETECTIONS = 300


@dataclass(frozen=True)
class Letterboxed:
    tensor: np.ndarray  # (1, 3, H, W) float32
    scale: float
    pad_x: int
    pad_y: int


@dataclass(frozen=True)
class Detection:
    cls: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def letterbox_image(image: Image.Image, spec: ManifestInput) -> Letterboxed:
    """Aspect-preserving scale, centered pad (floor), bilinear resample."""
    W, H = spec.width, spec.height
    w, h = image.size
    rgb = image.convert("RGB")
    if spec.letterbox.enabled:
        scale = min(W / w, H / h)
        nw, nh = round(w * scale), round(h * scale)
    else:
        scale, nw, nh = 1.0, W, H
    pad_x = (W - nw) // 2
    pad_y = (H - nh) // 2
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR, reducing_gap=None)
    canvas = Image.new("RGB", (W, H), (spec.letterbox.padValue,) * 3)
    canvas.paste(resized, (pad_x, pad_y))
    arr: np.ndarray = np.asarray(canvas, dtype=np.float32) * np.float32(spec.normalize.scale)
    if spec.normalize.mean is not None and spec.normalize.std is not None:
        arr = (arr - np.asarray(spec.normalize.mean, dtype=np.float32)) / np.asarray(
            spec.normalize.std, dtype=np.float32
        )
    tensor = arr.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
    return Letterboxed(np.ascontiguousarray(tensor), scale if spec.letterbox.enabled else 1.0, pad_x, pad_y)


def blank_frame(width: int = 640, height: int = 480) -> Image.Image:
    """Mid-grey, 4:3, so a square model input is letterboxed on two sides (the check's frame)."""
    return Image.new("RGB", (width, height), (114, 114, 114))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every row of `a` (n,4) against every row of `b` (m,4) -> (n,m)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    out: np.ndarray = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)
    return out


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy per-class NMS, highest score first; returns kept indexes."""
    keep: list[int] = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        suppressed = np.zeros(len(order), dtype=bool)
        for i, oi in enumerate(order):
            if suppressed[i]:
                continue
            keep.append(int(oi))
            if i + 1 >= len(order):
                break
            rest = order[i + 1 :]
            ious = iou_xyxy(boxes[oi][None], boxes[rest])[0]
            suppressed[i + 1 :] |= ious > iou_threshold
    keep.sort(key=lambda i: -float(scores[i]))
    return keep[:MAX_DETECTIONS]


def decode_yolov8(
    output: np.ndarray,
    lb: Letterboxed,
    source_size: tuple[int, int],
    *,
    min_confidence: float,
    iou_threshold: float,
) -> list[Detection]:
    """One (1, 4 + classes, N) head: rows [cx, cy, w, h, class scores...] in
    letterboxed input pixels, scores already 0..1, no objectness row."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"unexpected detection output shape {list(output.shape)}")
    preds = output[0]  # (4 + C, N)
    num_classes = preds.shape[0] - 4
    if num_classes < 1:
        raise ValueError(f"detection output has no class channels ({list(output.shape)})")
    scores_all = preds[4:]
    cls = scores_all.argmax(axis=0)
    score = scores_all.max(axis=0)
    mask = score >= min_confidence
    if not mask.any():
        return []
    cx, cy, bw, bh = preds[0][mask], preds[1][mask], preds[2][mask], preds[3][mask]
    cls, score = cls[mask], score[mask]
    sw, sh = source_size
    x1 = np.clip((cx - bw / 2 - lb.pad_x) / lb.scale, 0, sw - 1)
    y1 = np.clip((cy - bh / 2 - lb.pad_y) / lb.scale, 0, sh - 1)
    x2 = np.clip((cx + bw / 2 - lb.pad_x) / lb.scale, x1 + 1, sw)
    y2 = np.clip((cy + bh / 2 - lb.pad_y) / lb.scale, y1 + 1, sh)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    keep = nms_per_class(boxes, score.astype(np.float32), cls, iou_threshold)
    return [
        Detection(int(cls[i]), round(float(score[i]), 4), tuple(round(float(v), 1) for v in boxes[i]))  # type: ignore[arg-type]
        for i in keep
    ]


class OnnxDetector:
    """An ONNX Runtime session over a registry directory, decoding per the manifest."""

    def __init__(self, model_path: Path, manifest: ModelManifest, threads: int | None = None) -> None:
        import onnxruntime as ort  # heavy import, keep local

        self.manifest = manifest
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if manifest.postprocess.kind != "yolov8":
            raise ValueError(f'postprocess.kind "{manifest.postprocess.kind}" has no decoder in the forge')

    def run_raw(self, image: Image.Image) -> tuple[list[np.ndarray], Letterboxed]:
        lb = letterbox_image(image, self.manifest.input)
        outputs = self.session.run(None, {self.input_name: lb.tensor})
        return [np.asarray(o) for o in outputs], lb

    def detect(
        self, image: Image.Image, *, min_confidence: float, iou_threshold: float | None = None
    ) -> list[Detection]:
        outputs, lb = self.run_raw(image)
        head = next((o for o in outputs if o.ndim == 3), None)
        if head is None:
            raise ValueError(f"expected one rank-3 detection head, got {[list(o.shape) for o in outputs]}")
        return decode_yolov8(
            head,
            lb,
            image.size,
            min_confidence=min_confidence,
            iou_threshold=self.manifest.postprocess.nmsIouThreshold if iou_threshold is None else iou_threshold,
        )


def class_color(name: str) -> tuple[int, int, int]:
    """The engine's palette hash (`classColor`) so renders match the platform's overlays."""
    palette = [
        (230, 57, 70),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (231, 126, 34),
        (236, 240, 241),
    ]
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return palette[h % len(palette)]


def ort_version() -> str:
    try:
        import onnxruntime as ort

        return str(ort.__version__)
    except Exception:
        return "unknown version"


def dims_text(dims: list[Any]) -> str:
    return "[" + ",".join(str(d) for d in dims) + "]"


def is_finite(x: float) -> bool:
    return math.isfinite(x)
