"""`xmf dataset validate`: every image opens, every annotation references a
known class and lies inside its image, every image has a split, every class
has at least one validation instance (a warning below a floor, an error at
zero), and the header's counts match the files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .io import Dataset, load_dataset

DEFAULT_MIN_VAL_INSTANCES = 5


@dataclass
class ValidationReport:
    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    images: int = 0
    annotations: int = 0
    classes: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    val_instances: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        splits = ", ".join(f"{k} {v}" for k, v in self.split_counts.items())
        return (
            f"{self.images} images, {self.annotations} annotations, {len(self.classes)} classes "
            f"({', '.join(self.classes)}); splits: {splits}"
        )


def validate_dataset(
    root: Path,
    *,
    min_val_instances: int = DEFAULT_MIN_VAL_INSTANCES,
    open_images: bool = True,
    require_val: bool = True,
) -> ValidationReport:
    report = ValidationReport(root=Path(root))
    try:
        ds = load_dataset(root, require_splits=False)
    except (OSError, ValueError) as err:
        report.errors.append(str(err))
        return report
    return validate_loaded(
        ds, report, min_val_instances=min_val_instances, open_images=open_images, require_val=require_val
    )


def validate_loaded(
    ds: Dataset,
    report: ValidationReport | None = None,
    *,
    min_val_instances: int = DEFAULT_MIN_VAL_INSTANCES,
    open_images: bool = True,
    require_val: bool = True,
) -> ValidationReport:
    report = report or ValidationReport(root=ds.root)
    errors, warnings = report.errors, report.warnings
    classes = ds.header.classes
    report.classes = list(classes)
    report.images = len(ds.coco.images)
    report.annotations = len(ds.coco.annotations)

    # Categories are the 0-based index into classes, by id and by name.
    cats = {c.id: c.name for c in ds.coco.categories}
    expected = dict(enumerate(classes))
    if cats != expected:
        errors.append(
            f"categories do not match dataset.json classes: expected {expected}, got {cats} "
            "(ids are 0-based indexes into classes)"
        )
    if ds.header.images != len(ds.coco.images):
        errors.append(f"dataset.json says {ds.header.images} images, annotations.coco.json has {len(ds.coco.images)}")

    # Images
    images = {img.id: img for img in ds.coco.images}
    for img in ds.coco.images:
        p = ds.image_path(img.file_name)
        if not p.is_file():
            errors.append(f"image {img.id}: {img.file_name} not found")
            continue
        if open_images:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    fw, fh = im.size
            except Exception as err:
                errors.append(f"image {img.id}: {img.file_name} does not open ({err})")
                continue
            if (fw, fh) != (img.width, img.height):
                errors.append(f"image {img.id}: file is {fw}x{fh}, annotations say {img.width}x{img.height}")

    # Annotations
    for ann in ds.coco.annotations:
        owner = images.get(ann.image_id)
        if owner is None:
            errors.append(f"annotation {ann.id}: image {ann.image_id} does not exist")
            continue
        if ann.category_id not in cats:
            errors.append(f"annotation {ann.id}: category {ann.category_id} is not one of {sorted(cats)}")
        x, y, w, h = (float(v) for v in ann.bbox)
        if w <= 0 or h <= 0:
            errors.append(f"annotation {ann.id}: bbox {ann.bbox} has no area")
        elif x < -0.5 or y < -0.5 or x + w > img.width + 0.5 or y + h > img.height + 0.5:
            errors.append(
                f"annotation {ann.id}: bbox {ann.bbox} lies outside image {img.id} ({img.width}x{img.height})"
            )

    # Splits
    counts: Counter[str] = Counter()
    for img in ds.coco.images:
        s = ds.splits.get(img.id)
        if s is None:
            errors.append(f"image {img.id}: no split (run `xmf dataset split`)")
        else:
            counts[s] += 1
    for image_id in ds.splits:
        if image_id not in images:
            warnings.append(f"splits.json names image {image_id}, which is not in the dataset")
    report.split_counts = {k: counts.get(k, 0) for k in ("train", "val", "test")}
    header_counts = ds.header.splits.model_dump()
    if any(header_counts[k] != report.split_counts[k] for k in report.split_counts) and not any(
        "no split" in e for e in errors
    ):
        errors.append(f"dataset.json splits {header_counts} do not match splits.json {report.split_counts}")
    if counts.get("train", 0) == 0:
        errors.append("no image is in the train split")

    # Per-class validation instances
    val_instances: Counter[str] = Counter()
    for ann in ds.coco.annotations:
        if ds.splits.get(ann.image_id) == "val" and ann.category_id in cats:
            val_instances[cats[ann.category_id]] += 1
    report.val_instances = {c: val_instances.get(c, 0) for c in classes}
    if require_val:
        if counts.get("val", 0) == 0:
            errors.append("no image is in the val split")
        for c in classes:
            n = val_instances.get(c, 0)
            if n == 0:
                errors.append(f'class "{c}" has no instance in the val split')
            elif n < min_val_instances:
                warnings.append(f'class "{c}" has only {n} val instance(s) (floor {min_val_instances})')

    # Groups straddling splits
    by_group: dict[str, set[str]] = defaultdict(set)
    for img in ds.coco.images:
        if img.group and img.id in ds.splits:
            by_group[img.group].add(ds.splits[img.id])
    straddling = sorted(g for g, s in by_group.items() if len(s) > 1)
    if straddling:
        warnings.append(f"{len(straddling)} group(s) have images in more than one split: {', '.join(straddling[:5])}")

    # Classes without any annotation at all
    total: Counter[int] = Counter(a.category_id for a in ds.coco.annotations)
    for i, c in enumerate(classes):
        if total.get(i, 0) == 0:
            warnings.append(f'class "{c}" has no annotation in any split')

    return report
