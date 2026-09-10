"""`task: autolabel`: ONNX Runtime over the dataset's images with any
registry model, `minConfidence`, `classMap` (model class to dataset class),
writing the labels file the platform imports as `source: model`."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from .contracts import AutolabelJobSpec, LabelBox, LabelsFile, LabelsImage, LabelsSummary
from .contracts.labels import LabelsModel
from .dataset import Dataset
from .inference import OnnxDetector
from .seams import ResolvedModel
from .util import write_json


def run_assist(spec: AutolabelJobSpec, ds: Dataset, model: ResolvedModel, out_file: Path) -> LabelsFile:
    detector = OnnxDetector(model.model_path, model.manifest)
    names = model.manifest.classNames
    dataset_classes = set(ds.classes)
    class_map = dict(spec.classMap) or {n: n for n in names if n in dataset_classes}
    wanted = set(spec.images) if isinstance(spec.images, list) else None
    images: list[LabelsImage] = []
    per_class: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    boxes_total = 0
    for img in ds.coco.images:
        if wanted is not None and img.id not in wanted:
            continue
        with Image.open(ds.image_path(img.file_name)) as im:
            image = im.convert("RGB")
        dets = detector.detect(image, min_confidence=spec.minConfidence)
        boxes: list[LabelBox] = []
        for det in dets:
            model_cls = names[det.cls] if det.cls < len(names) else f"class_{det.cls}"
            target = class_map.get(model_cls)
            if target is None:
                unmapped[model_cls] += 1
                continue
            boxes.append(LabelBox(cls=target, modelCls=model_cls, confidence=det.confidence, xyxy=list(det.xyxy)))
            per_class[target] += 1
        boxes_total += len(boxes)
        images.append(LabelsImage(id=img.id, boxes=boxes))
    labels = LabelsFile(
        jobId=spec.jobId,
        model=LabelsModel(family=model.family, name=model.name, version=model.version),
        minConfidence=spec.minConfidence,
        classMap=class_map,
        images=images,
        summary=LabelsSummary(images=len(images), boxes=boxes_total, perClass=dict(per_class), unmapped=dict(unmapped)),
    )
    write_json(out_file, labels.to_json_dict())
    return labels
