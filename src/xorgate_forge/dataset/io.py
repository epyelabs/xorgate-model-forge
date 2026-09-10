"""Reading and writing a dataset v1 directory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..contracts import CocoFile, DatasetHeader, Split, SplitCounts, Splits, splits_adapter
from ..util import read_json, write_json

HEADER_FILE = "dataset.json"
COCO_FILE = "annotations.coco.json"
SPLITS_FILE = "splits.json"
IMAGES_DIR = "images"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class Dataset:
    root: Path
    header: DatasetHeader
    coco: CocoFile
    splits: Splits = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return self.header.classes

    def image_path(self, file_name: str) -> Path:
        return self.root / file_name

    def split_of(self, image_id: str) -> Split | None:
        return self.splits.get(image_id)

    def count_splits(self) -> SplitCounts:
        counts = {"train": 0, "val": 0, "test": 0}
        for image in self.coco.images:
            s = self.splits.get(image.id)
            if s is not None:
                counts[s] += 1
        return SplitCounts(**counts)

    def sync_header(self) -> None:
        """Recompute `images` and `splits` in the header from the files."""
        self.header.images = len(self.coco.images)
        self.header.splits = self.count_splits()


def load_dataset(root: Path, *, require_splits: bool = True) -> Dataset:
    root = Path(root)
    header_path = root / HEADER_FILE
    coco_path = root / COCO_FILE
    splits_path = root / SPLITS_FILE
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    if not header_path.is_file():
        raise FileNotFoundError(f"{header_path} not found (a dataset v1 directory has dataset.json)")
    if not coco_path.is_file():
        raise FileNotFoundError(f"{coco_path} not found")
    header = DatasetHeader.model_validate(read_json(header_path))
    coco = CocoFile.model_validate(read_json(coco_path))
    splits: Splits = {}
    if splits_path.is_file():
        splits = splits_adapter.validate_python(read_json(splits_path))
    elif require_splits:
        raise FileNotFoundError(f"{splits_path} not found (run `xmf dataset split`)")
    return Dataset(root, header, coco, splits)


def write_dataset(ds: Dataset) -> None:
    ds.sync_header()
    write_json(ds.root / HEADER_FILE, ds.header.to_json_dict())
    write_json(ds.root / COCO_FILE, ds.coco.model_dump(mode="json", exclude_none=True))
    write_json(ds.root / SPLITS_FILE, dict(sorted(ds.splits.items())))


def dataset_hash(root: Path) -> str:
    """sha256 over the three JSON files, in a fixed order; identifies a dataset version."""
    digest = hashlib.sha256()
    for name in (HEADER_FILE, COCO_FILE, SPLITS_FILE):
        p = Path(root) / name
        digest.update(name.encode())
        digest.update(b"\0")
        if p.is_file():
            digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
