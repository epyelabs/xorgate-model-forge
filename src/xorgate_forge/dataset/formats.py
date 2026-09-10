"""Interchange: `xmf dataset import --from coco|yolo|voc` into a dataset v1
directory and `xmf dataset export --to coco|yolo` out of one.

One internal representation (absolute `[x, y, w, h]` in the canonical COCO
file); YOLO's normalized `cx cy w h` is written with six decimals so a COCO
to YOLO to COCO round trip preserves boxes to well under a pixel (the golden
suite asserts 0.01 px on the fixture)."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..contracts import CocoAnnotation, CocoCategory, CocoFile, CocoImage, DatasetHeader, Split, SplitCounts, Splits
from ..util import read_json, write_json
from .io import COCO_FILE, HEADER_FILE, IMAGE_SUFFIXES, IMAGES_DIR, SPLITS_FILE, Dataset, write_dataset

SPLIT_DIRS = ("train", "val", "test")
_SPLIT_ALIASES: dict[str, Split] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


@dataclass
class ImportReport:
    images: int
    annotations: int
    classes: list[str]
    splits: dict[str, int]
    skipped: list[str]

    def render(self) -> str:
        s = ", ".join(f"{k} {v}" for k, v in self.splits.items())
        out = f"{self.images} images, {self.annotations} annotations, classes {self.classes}; splits: {s}"
        if self.skipped:
            out += f"; skipped {len(self.skipped)}: {'; '.join(self.skipped[:5])}"
        return out


def _copy_image(src: Path, dest_dir: Path, image_id: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    target = dest_dir / f"{image_id}{suffix}"
    if not target.exists():
        shutil.copy2(src, target)
    return f"{IMAGES_DIR}/{target.name}"


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _unique_id(stem: str, used: set[str]) -> str:
    image_id = stem
    n = 2
    while image_id in used:
        image_id = f"{stem}_{n}"
        n += 1
    used.add(image_id)
    return image_id


def _finish(
    root: Path, classes: list[str], images: list[CocoImage], anns: list[CocoAnnotation], splits: Splits
) -> Dataset:
    coco = CocoFile(
        info={"description": root.name},
        categories=[CocoCategory(id=i, name=c) for i, c in enumerate(classes)],
        images=images,
        annotations=anns,
    )
    header = DatasetHeader(classes=classes, images=len(images), splits=SplitCounts(train=0, val=0, test=0))
    ds = Dataset(root, header, coco, splits)
    write_dataset(ds)
    return ds


# --- COCO ----------------------------------------------------------------------


def import_coco(source: Path, root: Path, *, images_dir: Path | None = None, split: Split = "train") -> ImportReport:
    """A COCO detection JSON (any category ids, integer or string image ids)
    with its images next to it (or under `images_dir`). Categories are
    renumbered 0-based in the file's order; image ids become the file stems."""
    source, root = Path(source), Path(root)
    data = read_json(source)
    images_dir = images_dir or source.parent
    cats_in = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
    classes = [str(c["name"]) for c in cats_in]
    cat_index = {int(c["id"]): i for i, c in enumerate(cats_in)}
    images: list[CocoImage] = []
    anns: list[CocoAnnotation] = []
    splits: Splits = {}
    skipped: list[str] = []
    used: set[str] = set()
    id_map: dict[str, str] = {}
    image_splits: dict[str, Split] = {}
    for img in data.get("images", []):
        file_name = str(img["file_name"])
        src = images_dir / file_name
        if not src.is_file():
            src = images_dir / Path(file_name).name
        if not src.is_file():
            skipped.append(f"{file_name}: not found")
            continue
        image_id = _unique_id(str(img.get("xorgateImageId") or Path(file_name).stem), used)
        id_map[str(img["id"])] = image_id
        w, h = _image_size(src)
        rel = _copy_image(src, root / IMAGES_DIR, image_id)
        images.append(
            CocoImage(id=image_id, file_name=rel, width=w, height=h, group=img.get("group"), source=img.get("source"))
        )
        image_splits[image_id] = _SPLIT_ALIASES.get(str(img.get("split", "")).lower(), split)
    next_id = 1
    for ann in data.get("annotations", []):
        target_id = id_map.get(str(ann["image_id"]))
        if target_id is None:
            continue
        cat = cat_index.get(int(ann["category_id"]))
        if cat is None:
            skipped.append(f"annotation {ann.get('id')}: unknown category {ann['category_id']}")
            continue
        bx, by, bw, bh = (float(v) for v in ann["bbox"])
        anns.append(
            CocoAnnotation(
                id=next_id,
                image_id=target_id,
                category_id=cat,
                bbox=[bx, by, bw, bh],
                area=float(ann.get("area") or bw * bh),
                iscrowd=int(ann.get("iscrowd", 0)),
                source=ann.get("source"),
                confidence=ann.get("confidence"),
            )
        )
        next_id += 1
    splits = dict(image_splits)
    ds = _finish(root, classes, images, anns, splits)
    return ImportReport(len(images), len(anns), classes, ds.header.splits.model_dump(), skipped)


def export_coco(ds: Dataset, out: Path) -> Path:
    """A standard COCO file with integer image ids (sequential, the dataset's
    order) and a `xorgateImageId` on each image for the way back; categories
    keep their 0-based ids; images are copied next to it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {img.id: i + 1 for i, img in enumerate(ds.coco.images)}
    images = []
    for img in ds.coco.images:
        rel = _copy_image(ds.image_path(img.file_name), out / IMAGES_DIR, img.id)
        entry = {
            "id": index[img.id],
            "file_name": rel,
            "width": img.width,
            "height": img.height,
            "xorgateImageId": img.id,
        }
        if img.group:
            entry["group"] = img.group
        if img.source:
            entry["source"] = img.source
        if img.id in ds.splits:
            entry["split"] = ds.splits[img.id]
        images.append(entry)
    anns = []
    for ann in ds.coco.annotations:
        entry = {
            "id": ann.id,
            "image_id": index[ann.image_id],
            "category_id": ann.category_id,
            "bbox": [float(v) for v in ann.bbox],
            "area": float(ann.area if ann.area is not None else ann.bbox[2] * ann.bbox[3]),
            "iscrowd": ann.iscrowd,
        }
        if ann.source:
            entry["source"] = ann.source
        if ann.confidence is not None:
            entry["confidence"] = ann.confidence
        anns.append(entry)
    doc = {
        "info": {"description": ds.root.name, "categoryIds": "zero-based"},
        "categories": [{"id": c.id, "name": c.name} for c in ds.coco.categories],
        "images": images,
        "annotations": anns,
    }
    target = out / "annotations.json"
    write_json(target, doc)
    return target


# --- YOLO ----------------------------------------------------------------------


def _yolo_line(ann: CocoAnnotation, width: int, height: int) -> str:
    x, y, w, h = ann.bbox
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    return f"{ann.category_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def export_yolo(ds: Dataset, out: Path, *, link: bool = False) -> Path:
    """`<out>/images/<split>/<id>.jpg`, `<out>/labels/<split>/<id>.txt`,
    `<out>/data.yaml`. With `link`, images are symlinked instead of copied
    (what the trainer's temp layout does)."""
    out = Path(out)
    by_image: dict[str, list[CocoAnnotation]] = defaultdict(list)
    for ann in ds.coco.annotations:
        by_image[ann.image_id].append(ann)
    present: set[str] = set()
    for img in ds.coco.images:
        split = ds.splits.get(img.id, "train")
        present.add(split)
        img_dir = out / IMAGES_DIR / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        src = ds.image_path(img.file_name)
        target = img_dir / f"{img.id}{src.suffix.lower()}"
        if not target.exists():
            if link:
                target.symlink_to(src.resolve())
            else:
                shutil.copy2(src, target)
        lines = [_yolo_line(a, img.width, img.height) for a in by_image.get(img.id, [])]
        (lbl_dir / f"{img.id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(ds.classes))
    yaml_lines = [f"path: {out.resolve()}", "train: images/train"]
    yaml_lines.append("val: images/val" if "val" in present else "val: images/train")
    if "test" in present:
        yaml_lines.append("test: images/test")
    yaml_lines += [f"nc: {len(ds.classes)}", "names:", names, ""]
    data_yaml = out / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    return data_yaml


def _read_yaml_names(path: Path) -> list[str] | None:
    """Enough YAML for a `data.yaml`: `names:` as a list or an `i: name` map."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    names: dict[int, str] = {}
    inline: list[str] | None = None
    in_names = False
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("names:"):
            rest = line[len("names:") :].strip()
            if rest.startswith("["):
                inline = [s.strip().strip("'\"") for s in rest.strip("[]").split(",") if s.strip()]
                return inline
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t", "-")):
                break
            item = line.strip()
            if item.startswith("- "):
                names[len(names)] = item[2:].strip().strip("'\"")
            elif ":" in item:
                k, v = item.split(":", 1)
                names[int(k.strip())] = v.strip().strip("'\"")
    if names:
        return [names[i] for i in sorted(names)]
    return None


def import_yolo(source: Path, root: Path, *, classes: list[str] | None = None) -> ImportReport:
    """A YOLO layout: `images/<split>/` and `labels/<split>/` (or flat
    `images/` and `labels/`), class names from `data.yaml` or `--classes`."""
    source, root = Path(source), Path(root)
    if classes is None:
        for candidate in (source / "data.yaml", source / "dataset.yaml"):
            if candidate.is_file():
                classes = _read_yaml_names(candidate)
                break
    if not classes:
        raise ValueError("YOLO import needs class names: a data.yaml with `names:` next to the layout, or --classes")
    images_root = source / IMAGES_DIR
    labels_root = source / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(f"{images_root} not found")
    split_dirs = [d for d in images_root.iterdir() if d.is_dir() and _SPLIT_ALIASES.get(d.name.lower())]
    sources: list[tuple[Path, Split]] = []
    if split_dirs:
        for d in sorted(split_dirs):
            sources += [
                (p, _SPLIT_ALIASES[d.name.lower()]) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES
            ]
    else:
        sources = [(p, "train") for p in sorted(images_root.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    images: list[CocoImage] = []
    anns: list[CocoAnnotation] = []
    splits: Splits = {}
    skipped: list[str] = []
    used: set[str] = set()
    next_id = 1
    for path, image_split in sources:
        image_id = _unique_id(path.stem, used)
        w, h = _image_size(path)
        rel = _copy_image(path, root / IMAGES_DIR, image_id)
        images.append(CocoImage(id=image_id, file_name=rel, width=w, height=h))
        splits[image_id] = image_split
        label_path = labels_root / path.relative_to(images_root).with_suffix(".txt")
        if not label_path.is_file():
            continue
        for ln, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) < 5:
                if parts:
                    skipped.append(f"{label_path.name}:{ln}: expected `cls cx cy w h`")
                continue
            cls = int(parts[0])
            if cls < 0 or cls >= len(classes):
                skipped.append(f"{label_path.name}:{ln}: class {cls} outside 0..{len(classes) - 1}")
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            anns.append(
                CocoAnnotation(id=next_id, image_id=image_id, category_id=cls, bbox=[x, y, ww, hh], area=ww * hh)
            )
            next_id += 1
    _finish(root, list(classes), images, anns, splits)
    counts = {s: sum(1 for v in splits.values() if v == s) for s in SPLIT_DIRS}
    return ImportReport(len(images), len(anns), list(classes), counts, skipped)


# --- Pascal VOC ------------------------------------------------------------------


def import_voc(source: Path, root: Path, *, classes: list[str] | None = None, split: Split = "train") -> ImportReport:
    """Pascal VOC: `Annotations/*.xml` and `JPEGImages/` (or the XMLs next to
    the images). Classes are collected from the XMLs in sorted order unless
    `--classes` fixes the order."""
    source, root = Path(source), Path(root)
    ann_dir = source / "Annotations" if (source / "Annotations").is_dir() else source
    img_dir = source / "JPEGImages" if (source / "JPEGImages").is_dir() else source
    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"no VOC .xml files under {ann_dir}")
    parsed: list[tuple[Path, ET.Element]] = [(p, ET.parse(p).getroot()) for p in xmls]
    if classes is None:
        found: set[str] = set()
        for _, tree in parsed:
            for obj in tree.findall("object"):
                name = obj.findtext("name")
                if name:
                    found.add(name.strip())
        classes = sorted(found)
    index = {c: i for i, c in enumerate(classes)}
    images: list[CocoImage] = []
    anns: list[CocoAnnotation] = []
    splits: Splits = {}
    skipped: list[str] = []
    used: set[str] = set()
    next_id = 1
    for xml_path, tree in parsed:
        filename = tree.findtext("filename") or f"{xml_path.stem}.jpg"
        src = img_dir / filename
        if not src.is_file():
            candidates = [p for p in img_dir.glob(f"{xml_path.stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
            if not candidates:
                skipped.append(f"{xml_path.name}: image {filename} not found")
                continue
            src = candidates[0]
        image_id = _unique_id(src.stem, used)
        w, h = _image_size(src)
        rel = _copy_image(src, root / IMAGES_DIR, image_id)
        images.append(CocoImage(id=image_id, file_name=rel, width=w, height=h))
        splits[image_id] = split
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in index:
                skipped.append(f"{xml_path.name}: class {name!r} not in --classes")
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin, ymin, xmax, ymax = (float(box.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            anns.append(
                CocoAnnotation(
                    id=next_id,
                    image_id=image_id,
                    category_id=index[name],
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin) * (ymax - ymin),
                )
            )
            next_id += 1
    _finish(root, list(classes), images, anns, splits)
    counts = {s: sum(1 for v in splits.values() if v == s) for s in SPLIT_DIRS}
    return ImportReport(len(images), len(anns), list(classes), counts, skipped)


__all__ = [
    "COCO_FILE",
    "HEADER_FILE",
    "SPLITS_FILE",
    "ImportReport",
    "export_coco",
    "export_yolo",
    "import_coco",
    "import_voc",
    "import_yolo",
]
