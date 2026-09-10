"""`xmf dataset stats`: counts per split and class, box sizes, unlabeled images."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .io import Dataset


@dataclass
class DatasetStats:
    images: int
    annotations: int
    classes: list[str]
    images_per_split: dict[str, int]
    instances_per_class: dict[str, int]
    instances_per_class_per_split: dict[str, dict[str, int]]
    images_without_annotations: int
    groups: int
    box_area_fraction: dict[str, float] = field(default_factory=dict)
    annotation_sources: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "classes": self.classes,
            "imagesPerSplit": self.images_per_split,
            "instancesPerClass": self.instances_per_class,
            "instancesPerClassPerSplit": self.instances_per_class_per_split,
            "imagesWithoutAnnotations": self.images_without_annotations,
            "groups": self.groups,
            "boxAreaFraction": self.box_area_fraction,
            "annotationSources": self.annotation_sources,
        }

    def render(self) -> str:
        lines = [
            f"images {self.images}, annotations {self.annotations}, classes {len(self.classes)}, groups {self.groups}"
        ]
        lines.append("splits: " + ", ".join(f"{k} {v}" for k, v in self.images_per_split.items()))
        for c in self.classes:
            per = self.instances_per_class_per_split.get(c, {})
            lines.append(
                f"  {c:<20} {self.instances_per_class.get(c, 0):>6}   "
                + "  ".join(f"{s} {per.get(s, 0)}" for s in ("train", "val", "test"))
            )
        if self.images_without_annotations:
            lines.append(f"images without annotations: {self.images_without_annotations}")
        if self.box_area_fraction:
            b = self.box_area_fraction
            lines.append(f"box area fraction: min {b['min']:.4f}, median {b['median']:.4f}, max {b['max']:.4f}")
        if self.annotation_sources:
            lines.append("sources: " + ", ".join(f"{k} {v}" for k, v in sorted(self.annotation_sources.items())))
        return "\n".join(lines)


def compute_stats(ds: Dataset) -> DatasetStats:
    classes = ds.header.classes
    cats = {c.id: c.name for c in ds.coco.categories}
    images = {i.id: i for i in ds.coco.images}
    per_split: Counter[str] = Counter()
    for img in ds.coco.images:
        per_split[ds.splits.get(img.id, "unassigned")] += 1
    per_class: Counter[str] = Counter()
    per_class_split: dict[str, Counter[str]] = defaultdict(Counter)
    labeled: set[str] = set()
    fractions: list[float] = []
    sources: Counter[str] = Counter()
    for ann in ds.coco.annotations:
        name = cats.get(ann.category_id, f"class_{ann.category_id}")
        per_class[name] += 1
        per_class_split[name][ds.splits.get(ann.image_id, "unassigned")] += 1
        labeled.add(ann.image_id)
        sources[ann.source or "unknown"] += 1
        owner = images.get(ann.image_id)
        if owner is not None:
            fractions.append((ann.bbox[2] * ann.bbox[3]) / float(owner.width * owner.height))
    fractions.sort()
    box = {}
    if fractions:
        box = {"min": fractions[0], "median": fractions[len(fractions) // 2], "max": fractions[-1]}
    return DatasetStats(
        images=len(ds.coco.images),
        annotations=len(ds.coco.annotations),
        classes=list(classes),
        images_per_split={k: per_split.get(k, 0) for k in ("train", "val", "test")}
        | ({"unassigned": per_split["unassigned"]} if per_split.get("unassigned") else {}),
        instances_per_class={c: per_class.get(c, 0) for c in classes},
        instances_per_class_per_split={c: dict(per_class_split.get(c, Counter())) for c in classes},
        images_without_annotations=sum(1 for i in ds.coco.images if i.id not in labeled),
        groups=len({i.group for i in ds.coco.images if i.group}),
        box_area_fraction=box,
        annotation_sources=dict(sources),
    )
