"""Evaluate the exported ONNX over the validation split (design 3e): the
manifest's own pre and post-processing, then mAP50, mAP50-95 (COCO-style
101-point interpolation over IoU 0.50:0.05:0.95), precision and recall at
the confidence that maximizes F1 (the way Ultralytics reports them), the
per-class rows, a confusion matrix at that confidence, and up to 24 sample
renders with ground truth and predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contracts import Metrics, ModelManifest, PerClassMetrics
from .dataset import Dataset
from .inference import Detection, OnnxDetector, class_color, iou_xyxy
from .util import write_json

EVAL_MIN_CONFIDENCE = 0.001
PR_IOU = 0.5
CONFUSION_IOU = 0.45
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MAX_SAMPLES = 24
CONFIDENCE_GRID = np.linspace(0.05, 0.95, 91)


@dataclass
class EvalOutcome:
    metrics: Metrics
    metrics_file: Path
    confusion_file: Path
    samples: list[Path]
    val_images: int
    confidence: float


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO's 101-point interpolated average precision."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    points = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(points, mrec, mpre, right=0.0)))


def _match(preds: list[tuple[float, np.ndarray]], gts: np.ndarray, iou_t: float) -> np.ndarray:
    """Greedy matching by descending confidence for one image and class;
    returns a bool array of true positives aligned with `preds`."""
    tp = np.zeros(len(preds), dtype=bool)
    if len(gts) == 0 or not preds:
        return tp
    boxes = np.stack([p[1] for p in preds])
    ious = iou_xyxy(boxes, gts)
    taken = np.zeros(len(gts), dtype=bool)
    for i in range(len(preds)):
        row = ious[i].copy()
        row[taken] = -1
        j = int(row.argmax())
        if row[j] >= iou_t:
            taken[j] = True
            tp[i] = True
    return tp


def _draw_sample(
    image: Image.Image, gts: list[tuple[str, np.ndarray]], preds: list[Detection], names: list[str], confidence: float
) -> Image.Image:
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)
    for name, box in gts:
        draw.rectangle((float(box[0]), float(box[1]), float(box[2]), float(box[3])), outline=(46, 204, 113), width=2)
        draw.text((float(box[0]) + 2, max(0.0, float(box[1]) - 12)), f"gt {name}", fill=(46, 204, 113))
    for det in preds:
        if det.confidence < confidence:
            continue
        name = names[det.cls] if det.cls < len(names) else f"class_{det.cls}"
        color = class_color(name)
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 2, min(im.height - 12, y2 + 2)), f"{name} {det.confidence:.2f}", fill=color)
    return im


def evaluate_model(
    model_dir: Path, manifest: ModelManifest, ds: Dataset, out_dir: Path, *, split: str = "val"
) -> EvalOutcome:
    names = list(manifest.classNames)
    if names != ds.classes:
        raise ValueError(f"manifest classNames {names} do not match dataset classes {ds.classes}")
    detector = OnnxDetector(model_dir / "model.onnx", manifest)
    cats = {c.id: c.name for c in ds.coco.categories}
    gt_by_image: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for ann in ds.coco.annotations:
        x, y, w, h = ann.bbox
        gt_by_image[ann.image_id].append((ann.category_id, np.array([x, y, x + w, y + h], dtype=np.float32)))
    images = [img for img in ds.coco.images if ds.splits.get(img.id) == split]
    nc = len(names)

    # Pass 1: detections at the evaluation floor, ground truth per image.
    dets_by_image: list[list[Detection]] = []
    gts_by_image: list[list[tuple[int, np.ndarray]]] = []
    for img in images:
        with Image.open(ds.image_path(img.file_name)) as im:
            image = im.convert("RGB")
        dets_by_image.append(detector.detect(image, min_confidence=EVAL_MIN_CONFIDENCE))
        gts_by_image.append(gt_by_image.get(img.id, []))

    # Per class: confidence-sorted true-positive flags per IoU threshold, and the flag at IoU 0.5 for P/R.
    scores: dict[int, list[float]] = defaultdict(list)
    tps: dict[int, list[np.ndarray]] = defaultdict(list)
    gt_count = np.zeros(nc, dtype=int)
    for dets, gts in zip(dets_by_image, gts_by_image):
        for c, _ in gts:
            gt_count[c] += 1
        for c in range(nc):
            c_gts = (
                np.stack([b for cc, b in gts if cc == c])
                if any(cc == c for cc, _ in gts)
                else np.zeros((0, 4), np.float32)
            )
            c_preds = sorted(
                ((d.confidence, np.array(d.xyxy, np.float32)) for d in dets if d.cls == c), key=lambda p: -p[0]
            )
            if not c_preds:
                continue
            per_t = np.stack([_match(c_preds, c_gts, float(t)) for t in IOU_THRESHOLDS], axis=1)  # (n, 10)
            for k, (conf, _) in enumerate(c_preds):
                scores[c].append(conf)
                tps[c].append(per_t[k])

    # The reporting confidence: the grid point that maximizes F1 over every class at IoU 0.5.
    all_conf = (
        np.concatenate([np.asarray(scores[c]) for c in range(nc) if scores[c]]) if any(scores.values()) else np.zeros(0)
    )
    all_tp = (
        np.concatenate([np.stack(tps[c])[:, 0] for c in range(nc) if tps[c]])
        if any(tps.values())
        else np.zeros(0, dtype=bool)
    )
    total_gt = int(gt_count.sum())
    best_conf, best_f1 = 0.25, -1.0
    for c_thr in CONFIDENCE_GRID:
        kept = all_conf >= c_thr
        tp = int(all_tp[kept].sum())
        fp = int(kept.sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 + 1e-9:
            best_conf, best_f1 = float(c_thr), f1
    confidence = round(best_conf, 2)

    per_class: list[PerClassMetrics] = []
    ap50_all: list[float] = []
    ap_all: list[float] = []
    sum_tp = sum_fp = 0
    for c in range(nc):
        n_gt = int(gt_count[c])
        ap50, ap = 0.0, 0.0
        tp_c = fp_c = 0
        if scores[c]:
            conf_c = np.asarray(scores[c])
            order = np.argsort(-conf_c, kind="stable")
            tp_mat = np.stack(tps[c])[order]  # (n, 10)
            if n_gt > 0:
                aps = []
                for t in range(len(IOU_THRESHOLDS)):
                    tp_cum = np.cumsum(tp_mat[:, t])
                    fp_cum = np.cumsum(~tp_mat[:, t])
                    recall_curve = tp_cum / n_gt
                    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                    aps.append(_ap_from_pr(recall_curve, precision_curve))
                ap50, ap = aps[0], float(np.mean(aps))
            kept = conf_c[order] >= confidence
            tp_c = int(tp_mat[kept, 0].sum())
            fp_c = int(kept.sum()) - tp_c
        sum_tp += tp_c
        sum_fp += fp_c
        if n_gt > 0:
            ap50_all.append(ap50)
            ap_all.append(ap)
        per_class.append(
            PerClassMetrics(
                name=names[c],
                mAP50=round(ap50, 4),
                mAP50_95=round(ap, 4),
                precision=round(tp_c / (tp_c + fp_c), 4) if tp_c + fp_c else 0.0,
                recall=round(tp_c / n_gt, 4) if n_gt else 0.0,
                instances=n_gt,
            )
        )
    metrics = Metrics(
        mAP50=round(float(np.mean(ap50_all)) if ap50_all else 0.0, 4),
        mAP50_95=round(float(np.mean(ap_all)) if ap_all else 0.0, 4),
        precision=round(sum_tp / (sum_tp + sum_fp), 4) if sum_tp + sum_fp else 0.0,
        recall=round(sum_tp / total_gt, 4) if total_gt else 0.0,
        perClass=per_class,
    )

    # Confusion matrix at the reporting confidence (rows predicted, columns true, last = background).
    confusion = np.zeros((nc + 1, nc + 1), dtype=int)
    for dets, gts in zip(dets_by_image, gts_by_image):
        strong = sorted((d for d in dets if d.confidence >= confidence), key=lambda d: -d.confidence)
        gt_boxes = np.stack([b for _, b in gts]) if gts else np.zeros((0, 4), np.float32)
        gt_cls = [c for c, _ in gts]
        matched: set[int] = set()
        if strong:
            det_boxes = np.stack([np.array(d.xyxy, np.float32) for d in strong])
            ious = iou_xyxy(det_boxes, gt_boxes) if gts else np.zeros((len(strong), 0), np.float32)
            for i, d in enumerate(strong):
                best_j, best_iou = -1, CONFUSION_IOU
                for j in range(len(gts)):
                    if j not in matched and ious[i, j] >= best_iou:
                        best_j, best_iou = j, float(ious[i, j])
                if best_j >= 0:
                    matched.add(best_j)
                    confusion[d.cls, gt_cls[best_j]] += 1
                else:
                    confusion[d.cls, nc] += 1
        for j, c in enumerate(gt_cls):
            if j not in matched:
                confusion[nc, c] += 1

    # Sample renders.
    samples_dir = out_dir / "samples"
    samples: list[Path] = []
    for idx, (img, dets, gts) in enumerate(zip(images, dets_by_image, gts_by_image)):
        if idx >= MAX_SAMPLES:
            break
        samples_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(ds.image_path(img.file_name)) as im:
            render = _draw_sample(im, [(cats[c], b) for c, b in gts], dets, names, confidence)
        target = samples_dir / f"{idx}.jpg"
        render.save(target, quality=85)
        samples.append(target)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    detail: dict[str, Any] = {
        **metrics.to_json_dict(),
        "split": split,
        "images": len(images),
        "iouThresholds": [round(float(t), 2) for t in IOU_THRESHOLDS],
        "reportingConfidence": confidence,
        "reportingIou": PR_IOU,
        "model": {"family": manifest.family, "name": manifest.name, "version": manifest.version},
    }
    write_json(metrics_file, detail)
    confusion_file = out_dir / "confusion.json"
    write_json(
        confusion_file,
        {
            "classes": [*names, "background"],
            "rows": "predicted",
            "columns": "true",
            "confidence": confidence,
            "iou": CONFUSION_IOU,
            "matrix": confusion.tolist(),
        },
    )
    return EvalOutcome(metrics, metrics_file, confusion_file, samples, len(images), confidence)
