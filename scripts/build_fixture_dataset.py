#!/usr/bin/env python3
"""Build the golden fixture dataset (design 3g) from the engine's drive.mp4:
frames sampled at a fixed stride, pre-labeled with the pulled
`detection/yolov8n` for `car` and `truck`, grouped by five-second bucket,
split by group under seed 0, written as dataset v1 into fixtures/ds.

The result was hand-checked once (a contact sheet of every frame with its
boxes, see the M1 STATUS entry) and is committed; re-running this script
regenerates it deterministically from the same video and model.

    uv run python scripts/build_fixture_dataset.py [--stride 6] [--min-confidence 0.4]
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from xorgate_forge.contracts import CocoAnnotation, CocoCategory, CocoFile, CocoImage, DatasetHeader, SplitCounts
from xorgate_forge.dataset import Dataset, assign_splits, write_dataset
from xorgate_forge.hosts.local import LocalModelSource
from xorgate_forge.inference import OnnxDetector, class_color

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "fixtures" / "video" / "drive.mp4"
MODELS = ROOT / "fixtures" / "models"
OUT = ROOT / "fixtures" / "ds"
CLASSES = ["car", "truck"]
GROUP_SECONDS = 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=6, help="keep every Nth frame (15 fps video: 6 -> 2.5 fps)")
    ap.add_argument("--min-confidence", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--contact-sheet", type=Path, help="write a contact sheet of every frame with its boxes")
    args = ap.parse_args()

    if not VIDEO.is_file():
        raise SystemExit(f"{VIDEO} missing: run `make fixtures`")
    model = LocalModelSource(roots=[MODELS]).fetch("detection", "yolov8n")
    detector = OnnxDetector(model.model_path, model.manifest)
    names = model.manifest.classNames

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    images: list[CocoImage] = []
    anns: list[CocoAnnotation] = []
    renders: list[Image.Image] = []
    index = 0
    next_ann = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride == 0:
            ts = index / fps
            image_id = f"drive_{index:04d}"
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rel = f"images/{image_id}.jpg"
            rgb.save(out / rel, quality=85, subsampling=1)
            with Image.open(out / rel) as saved:
                w, h = saved.size
                decoded = saved.convert("RGB")
            images.append(
                CocoImage(
                    id=image_id,
                    file_name=rel,
                    width=w,
                    height=h,
                    group=f"drive:{int(ts // GROUP_SECONDS)}",
                    source={"kind": "video", "file": "fixtures/video/drive.mp4", "frame": index, "ts": round(ts, 3)},
                )
            )
            draw = ImageDraw.Draw(decoded)
            for det in detector.detect(decoded, min_confidence=args.min_confidence):
                cls_name = names[det.cls]
                if cls_name not in CLASSES:
                    continue
                x1, y1, x2, y2 = det.xyxy
                anns.append(
                    CocoAnnotation(
                        id=next_ann,
                        image_id=image_id,
                        category_id=CLASSES.index(cls_name),
                        bbox=[x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
                        area=round((x2 - x1) * (y2 - y1), 1),
                        iscrowd=0,
                        source="model",
                        confidence=det.confidence,
                    )
                )
                next_ann += 1
                draw.rectangle((x1, y1, x2, y2), outline=class_color(cls_name), width=3)
                draw.text((x1 + 3, y1 + 3), f"{cls_name} {det.confidence:.2f}", fill=class_color(cls_name))
            draw.text((8, 8), image_id, fill=(255, 255, 0))
            renders.append(decoded.resize((320, 180)))
        index += 1
    cap.release()

    coco = CocoFile(
        info={
            "description": "fixture-vehicles: frames of the engine's drive.mp4, pre-labeled with detection/yolov8n v1"
        },
        categories=[CocoCategory(id=i, name=c) for i, c in enumerate(CLASSES)],
        images=images,
        annotations=anns,
    )
    header = DatasetHeader(
        classes=CLASSES,
        images=len(images),
        splits=SplitCounts(train=0, val=0, test=0),
        source={
            "platform": "xorgate-model-forge",
            "datasetName": "fixture-vehicles",
            "video": "fixtures/video/drive.mp4",
        },  # type: ignore[arg-type]
    )
    ds = Dataset(out, header, coco, assign_splits(coco, train=0.8, val=0.2, test=0.0, seed=0))
    write_dataset(ds)
    per_class = Counter(CLASSES[a.category_id] for a in anns)
    per_split_class: Counter[str] = Counter(f"{ds.splits[a.image_id]}/{CLASSES[a.category_id]}" for a in anns)
    print(f"{len(images)} frames (stride {args.stride}, {fps:.2f} fps), {len(anns)} boxes {dict(per_class)}")
    print(f"splits {ds.header.splits.model_dump()}, per split/class {dict(sorted(per_split_class.items()))}")

    if args.contact_sheet and renders:
        cols = 6
        rows = (len(renders) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 180), (0, 0, 0))
        for i, r in enumerate(renders):
            sheet.paste(r, ((i % cols) * 320, (i // cols) * 180))
        sheet.save(args.contact_sheet, quality=80)
        print(f"contact sheet -> {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
