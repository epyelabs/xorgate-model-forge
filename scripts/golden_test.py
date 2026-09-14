#!/usr/bin/env python3
"""The repo's regression gate (design 3g), in the engine's golden-test
style: every case prints one `ok` / `FAIL` line, the suite prints its
duration and exits 1 when anything failed. Driven through the real `xmf`
CLI (and the real `npx xwe` for the two engine cases), never through
imports where a user would type a command.

    make test            # pulls the fixtures first
    XMF_GOLDEN_SKIP_TRAIN=1 uv run python scripts/golden_test.py   # contracts and dataset cases only
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
DS = FIXTURES / "ds"
OUT_ROOT = Path(tempfile.mkdtemp(prefix="xmf-golden-"))
SKIP_TRAIN = os.environ.get("XMF_GOLDEN_SKIP_TRAIN") == "1"
NODE = shutil.which("node")
XWE = ROOT / "node_modules" / ".bin" / "xwe"
# The engine spawns `ffmpeg` from PATH or FFMPEG_PATH; the dev dependency ffmpeg-static covers machines without one.
FFMPEG_STATIC = ROOT / "node_modules" / "ffmpeg-static" / "ffmpeg"
XWE_ENV = {"FFMPEG_PATH": str(FFMPEG_STATIC)} if not shutil.which("ffmpeg") and FFMPEG_STATIC.exists() else {}

failed = False
started = time.monotonic()


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)
    if not ok:
        failed = True


def run(
    args: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT, timeout: int = 1800
) -> subprocess.CompletedProcess[str]:
    t0 = time.monotonic()
    proc = subprocess.run(
        args,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    proc.seconds = time.monotonic() - t0  # type: ignore[attr-defined]
    return proc


def xmf(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return run([sys.executable, "-m", "xorgate_forge.cli.main", *args], **kw)


def xwe(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    kw["env"] = {**XWE_ENV, **(kw.get("env") or {})}
    return run([str(XWE), *args], **kw)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def last_line(s: str) -> str:
    lines = [ln for ln in s.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# --- contracts: schemas current, examples validate ---------------------------------
print("--- contracts")


def _case_1() -> None:
    from xorgate_forge.contracts import (
        CocoFile,
        DatasetHeader,
        JobResult,
        LabelsFile,
        ModelManifest,
        parse_job_spec,
        progress_adapter,
        splits_adapter,
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from emit_schemas import emit

    fresh = OUT_ROOT / "schemas"
    emitted = emit(fresh)
    drift = [p.name for p in emitted if (ROOT / "contracts" / p.name).read_text() != p.read_text()]
    check("contracts/*.schema.json match a fresh emission (make schemas)", not drift, ", ".join(drift))
    if NODE:
        vend = OUT_ROOT / "model-manifest.schema.json"
        r = run([NODE, str(ROOT / "scripts" / "vendor-manifest-schema.mjs"), str(vend)])
        same = r.returncode == 0 and vend.read_text() == (ROOT / "contracts" / "model-manifest.schema.json").read_text()
        check("contracts/model-manifest.schema.json matches the engine's schema (vendored)", same, last_line(r.stderr))
    ex = ROOT / "contracts" / "examples"
    try:
        DatasetHeader.model_validate(read_json(ex / "dataset.json"))
        CocoFile.model_validate(read_json(ex / "annotations.coco.json"))
        splits_adapter.validate_python(read_json(ex / "splits.json"))
        parse_job_spec(read_json(ex / "job-train.json"))
        parse_job_spec(read_json(ex / "job-autolabel.json"))
        JobResult.model_validate(read_json(ex / "result-train.json"))
        JobResult.model_validate(read_json(ex / "result-autolabel.json"))
        for line in (ex / "progress.jsonl").read_text().splitlines():
            progress_adapter.validate_python(json.loads(line))
        m = ModelManifest.model_validate(read_json(ex / "manifest-with-provenance.json"))
        LabelsFile.model_validate(read_json(ex / "autolabel-output.json"))
        check(
            "the ten example documents validate against the models",
            m.provenance is not None and m.provenance.recipe == "yolov8",
        )
    except Exception as err:
        check("the ten example documents validate against the models", False, f"{type(err).__name__}: {err}")
    bad = parse_job_spec
    try:
        bad({"specVersion": 1, "jobId": "x", "task": "train", "recipe": "yolov8", "base": {"kind": "pretrained"}})
        check("a train spec without output and base.ref is rejected", False)
    except Exception as err:
        check("a train spec without output and base.ref is rejected", "output" in str(err) or "ref" in str(err))


_case_1()


# --- dataset tools ------------------------------------------------------------------
print("--- dataset")


def _case_2() -> None:
    r = xmf(["dataset", "validate", str(DS)])
    check(
        "dataset validate fixtures/ds exits 0 and prints OK",
        r.returncode == 0 and r.stdout.strip().splitlines()[-2].startswith("OK: 63 images, 108 annotations, 2 classes"),
        last_line(r.stdout) or last_line(r.stderr),
    )
    check("dataset validate warns about the thin truck val count", "truck" in r.stdout and "val instance" in r.stdout)

    a = OUT_ROOT / "split-a"
    b = OUT_ROOT / "split-b"
    shutil.copytree(DS, a)
    shutil.copytree(DS, b)
    ra = xmf(["dataset", "split", str(a), "--seed", "0"])
    rb = xmf(["dataset", "split", str(b), "--seed", "0"])
    same = ra.returncode == 0 and rb.returncode == 0 and read_json(a / "splits.json") == read_json(b / "splits.json")
    check("dataset split is deterministic under a seed", same, last_line(ra.stdout))
    check(
        "dataset split seed 0 reproduces the committed splits.json",
        read_json(a / "splits.json") == read_json(DS / "splits.json"),
    )
    # Five groups, so many seeds coincide; some seed in 1..20 must deal them differently.
    differing = []
    for seed in range(1, 21):
        rc = xmf(["dataset", "split", str(b), "--seed", str(seed)])
        if rc.returncode == 0 and read_json(b / "splits.json") != read_json(DS / "splits.json"):
            differing.append(seed)
    check("another seed changes the assignment", bool(differing), f"seeds that differ from 0: {differing}")
    groups: dict[str, set[str]] = {}
    coco = read_json(a / "annotations.coco.json")
    sp = read_json(a / "splits.json")
    for img in coco["images"]:
        groups.setdefault(img["group"], set()).add(sp[img["id"]])
    check(
        "no group straddles two splits",
        all(len(s) == 1 for s in groups.values()),
        str({g: s for g, s in groups.items() if len(s) > 1}),
    )
    keep = {k: v for k, v in read_json(DS / "splits.json").items()}
    kept_ids = list(keep)[:10]
    keep_file = OUT_ROOT / "keep.json"
    keep_file.write_text(json.dumps({k: ("test" if keep[k] == "train" else keep[k]) for k in kept_ids}))
    rk = xmf(["dataset", "split", str(b), "--seed", "0", "--keep", str(keep_file)])
    sk = read_json(b / "splits.json")
    check(
        "--keep preserves the previous assignment of the named images",
        rk.returncode == 0 and all(sk[k] == ("test" if keep[k] == "train" else keep[k]) for k in kept_ids),
        last_line(rk.stderr),
    )

    # Round trips: COCO -> YOLO -> COCO preserves boxes to the pixel (0.01 px here).
    yolo_dir = OUT_ROOT / "yolo"
    r1 = xmf(["dataset", "export", str(DS), "--to", "yolo", "--out", str(yolo_dir)])
    back = OUT_ROOT / "from-yolo"
    r2 = xmf(["dataset", "import", str(yolo_dir), "--from", "yolo", "--out", str(back)])
    orig = read_json(DS / "annotations.coco.json")
    got = (
        read_json(back / "annotations.coco.json")
        if (back / "annotations.coco.json").exists()
        else {"annotations": [], "images": []}
    )

    def boxes_by_image(doc: dict[str, Any]) -> dict[str, list[tuple[int, list[float]]]]:
        out: dict[str, list[tuple[int, list[float]]]] = {}
        for ann in doc["annotations"]:
            out.setdefault(str(ann["image_id"]), []).append((ann["category_id"], [float(v) for v in ann["bbox"]]))
        for v in out.values():
            v.sort(key=lambda t: (t[0], t[1]))
        return out

    def max_delta(x: dict[str, Any], y: dict[str, Any]) -> float:
        bx, by = boxes_by_image(x), boxes_by_image(y)
        if bx.keys() != by.keys() or any(len(bx[k]) != len(by[k]) for k in bx):
            return float("inf")
        worst = 0.0
        for k in bx:
            for (ca, a), (cb, b) in zip(bx[k], by[k]):
                if ca != cb:
                    return float("inf")
                worst = max(worst, max(abs(p - q) for p, q in zip(a, b)))
        return worst

    delta = max_delta(orig, got)
    check(
        "COCO -> YOLO -> COCO preserves every box within 0.01 px",
        r1.returncode == 0 and r2.returncode == 0 and delta <= 0.01,
        f"max delta {delta}",
    )
    check(
        "YOLO import keeps the split from the layout",
        (back / "splits.json").exists() and read_json(back / "splits.json") == read_json(DS / "splits.json"),
    )
    coco_dir = OUT_ROOT / "coco"
    r3 = xmf(["dataset", "export", str(DS), "--to", "coco", "--out", str(coco_dir)])
    back2 = OUT_ROOT / "from-coco"
    r4 = xmf(["dataset", "import", str(coco_dir / "annotations.json"), "--from", "coco", "--out", str(back2)])
    got2 = (
        read_json(back2 / "annotations.coco.json")
        if (back2 / "annotations.coco.json").exists()
        else {"annotations": [], "images": []}
    )
    check(
        "COCO export -> import is exact (ids, classes, boxes, splits)",
        r3.returncode == 0
        and r4.returncode == 0
        and max_delta(orig, got2) == 0.0
        and read_json(back2 / "splits.json") == read_json(DS / "splits.json"),
        last_line(r4.stderr),
    )
    rv = xmf(["dataset", "validate", str(back2)])
    check("the re-imported dataset validates", rv.returncode == 0, last_line(rv.stderr))

    # VOC: synthesize from the fixture, import, compare.
    voc = OUT_ROOT / "voc"
    (voc / "Annotations").mkdir(parents=True)
    (voc / "JPEGImages").mkdir()
    names = ["car", "truck"]
    by_img: dict[str, list[dict[str, Any]]] = {}
    for ann in orig["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)
    for img in orig["images"]:
        shutil.copy(DS / img["file_name"], voc / "JPEGImages" / f"{img['id']}.jpg")
        objs = "".join(
            f"<object><name>{names[a['category_id']]}</name><bndbox><xmin>{a['bbox'][0]}</xmin><ymin>{a['bbox'][1]}</ymin>"
            f"<xmax>{a['bbox'][0] + a['bbox'][2]}</xmax><ymax>{a['bbox'][1] + a['bbox'][3]}</ymax></bndbox></object>"
            for a in by_img.get(img["id"], [])
        )
        (voc / "Annotations" / f"{img['id']}.xml").write_text(
            f"<annotation><filename>{img['id']}.jpg</filename><size><width>{img['width']}</width><height>{img['height']}</height></size>{objs}</annotation>"
        )
    back3 = OUT_ROOT / "from-voc"
    r5 = xmf(["dataset", "import", str(voc), "--from", "voc", "--out", str(back3), "--classes", "car,truck"])
    got3 = (
        read_json(back3 / "annotations.coco.json")
        if (back3 / "annotations.coco.json").exists()
        else {"annotations": [], "images": []}
    )
    d3 = max_delta(orig, got3)
    check(
        "VOC import preserves boxes within 0.01 px",
        r5.returncode == 0 and d3 <= 0.01,
        f"max delta {d3}; {last_line(r5.stderr)}",
    )
    rs = xmf(["dataset", "stats", str(DS), "--json"])
    try:
        stats = json.loads(rs.stdout)
        check(
            "dataset stats --json reports the fixture",
            stats["images"] == 63 and stats["instancesPerClass"] == {"car": 92, "truck": 16} and stats["groups"] == 5,
            json.dumps(stats.get("imagesPerSplit")),
        )
    except Exception as err:
        check("dataset stats --json reports the fixture", False, str(err))


_case_2()


# --- CLI: doctor, recipes, validate, exit codes -------------------------------------
print("--- cli")


def _case_3() -> None:
    r = xmf(["doctor"])
    check(
        "doctor exits 0 and names the yolov8 recipe with its license",
        r.returncode == 0 and "yolov8" in r.stdout and "AGPL-3.0" in r.stdout and "onnxruntime  ok" in r.stdout,
        last_line(r.stderr),
    )
    r = xmf(["recipes"])
    check("recipes lists yolov8 with AGPL-3.0", r.returncode == 0 and "yolov8" in r.stdout and "AGPL-3.0" in r.stdout)
    r = xmf(["recipes", "--json"])
    check(
        "recipes --json is a JSON array with presets",
        r.returncode == 0 and json.loads(r.stdout)[0]["presets"]["fast"]["epochs"] == 50,
    )
    r = xmf(["--log-level", "silent", "recipes"])
    check("--log-level silent leaves stderr empty", r.returncode == 0 and r.stderr == "")
    r = xmf(["validate", str(FIXTURES / "specs" / "quick.json"), "--dataset", str(DS)])
    check(
        "validate quick.json exits 0 and says what the job needs",
        r.returncode == 0 and "OK: job quick can run" in r.stdout and "recipe yolov8 (AGPL-3.0)" in r.stdout,
        last_line(r.stderr),
    )
    r = xmf(["validate", str(FIXTURES / "specs" / "bad.json"), "--dataset", str(DS)])
    check(
        "validate on a bad spec exits 2 naming the field",
        r.returncode == 2 and "bad.json" in r.stderr and "output" in r.stderr,
        last_line(r.stderr),
    )
    r = xmf(["run", str(FIXTURES / "specs" / "bad.json"), "--dataset", str(DS), "--out", str(OUT_ROOT / "bad")])
    check(
        "run on a bad spec exits 2 and writes nothing",
        r.returncode == 2 and not (OUT_ROOT / "bad").exists(),
        last_line(r.stderr),
    )
    r = xmf(["run", str(FIXTURES / "specs" / "quick.json"), "--out", str(OUT_ROOT / "noargs")])
    check("run without --dataset exits 2", r.returncode == 2 and "--dataset" in r.stderr, last_line(r.stderr))
    r = xmf(
        [
            "validate",
            str(FIXTURES / "specs" / "missing-class.json"),
            "--dataset",
            str(DS),
            "--models-dir",
            str(FIXTURES / "models"),
        ]
    )
    check(
        "validate on the missing-class spec exits 1 naming the class",
        r.returncode == 1 and "forklift" in r.stderr,
        last_line(r.stderr),
    )
    r = xmf(["nope"])
    check("an unknown command exits 2 with the usage block", r.returncode == 2 and "usage:" in r.stderr)
    r = xmf(["check", str(FIXTURES / "models" / "detection" / "yolov8n" / "1"), "--no-fill"])
    check(
        "check on the pulled yolov8n prints four ok lines",
        r.returncode == 0 and [ln.split()[1] for ln in r.stdout.strip().splitlines()] == ["ok"] * 4,
        last_line(r.stdout),
    )
    wrong = OUT_ROOT / "wrong" / "detection" / "yolov8n-two" / "1"
    wrong.mkdir(parents=True)
    shutil.copy(FIXTURES / "models" / "detection" / "yolov8n" / "1" / "model.onnx", wrong / "model.onnx")
    m = read_json(FIXTURES / "models" / "detection" / "yolov8n" / "1" / "manifest.json")
    m["name"] = "yolov8n-two"
    m["classNames"] = ["forklift", "pallet"]
    (wrong / "manifest.json").write_text(json.dumps(m))
    r = xmf(["check", str(wrong)])
    expect = 'output     fail "output0" [1,84,8400] does not match yolov8 with 2 classes: expected 4 + 2 = 6 rows, got 84 (a model with 80 classes?)'
    check(
        "a class-count mismatch fails the output line with the engine's wording",
        r.returncode == 1 and expect in r.stdout,
        last_line(r.stdout),
    )
    if NODE and XWE.exists():
        rx = xwe(["models", "check", str(wrong)])
        check(
            "xwe agrees on the class-count mismatch line",
            rx.returncode == 1 and expect in rx.stdout,
            last_line(rx.stdout),
        )


_case_3()


# --- failure path: a spec naming a missing class ------------------------------------
print("--- failure paths")


def _case_4() -> None:
    out = OUT_ROOT / "fail"
    r = xmf(
        [
            "run",
            str(FIXTURES / "specs" / "missing-class.json"),
            "--dataset",
            str(DS),
            "--models-dir",
            str(FIXTURES / "models"),
            "--out",
            str(out),
        ]
    )
    res_path = out / "missing-class" / "result.json"
    res = read_json(res_path) if res_path.exists() else {}
    check(
        "a spec naming a missing class exits 1 with result.json written",
        r.returncode == 1 and res.get("status") == "failed",
        last_line(r.stderr),
    )
    check(
        "its error names phase prepare and the class",
        res.get("error", {}).get("phase") == "prepare" and "forklift" in res.get("error", {}).get("message", ""),
        json.dumps(res.get("error")),
    )
    prog = (
        [json.loads(ln) for ln in (out / "missing-class" / "progress.jsonl").read_text().splitlines()]
        if (out / "missing-class" / "progress.jsonl").exists()
        else []
    )
    check(
        "progress.jsonl ends with status failed, phase done",
        bool(prog) and prog[-1]["t"] == "status" and prog[-1]["status"] == "failed" and prog[-1]["phase"] == "done",
    )
    # A train job over a dataset with a class lacking validation instances.
    thin = OUT_ROOT / "thin-ds"
    shutil.copytree(DS, thin)
    sp = read_json(thin / "splits.json")
    coco = read_json(thin / "annotations.coco.json")
    truck_val = {a["image_id"] for a in coco["annotations"] if a["category_id"] == 1 and sp[a["image_id"]] == "val"}
    for image_id in truck_val:
        sp[image_id] = "train"
    (thin / "splits.json").write_text(json.dumps(sp))
    hdr = read_json(thin / "dataset.json")
    hdr["splits"] = {
        "train": sum(1 for v in sp.values() if v == "train"),
        "val": sum(1 for v in sp.values() if v == "val"),
        "test": 0,
    }
    (thin / "dataset.json").write_text(json.dumps(hdr))
    r = xmf(["run", str(FIXTURES / "specs" / "quick.json"), "--dataset", str(thin), "--out", str(out)])
    res = read_json(out / "quick" / "result.json") if (out / "quick" / "result.json").exists() else {}
    check(
        "a train job over a class with no val instance fails in prepare naming the class",
        r.returncode == 1
        and res.get("error", {}).get("phase") == "prepare"
        and '"truck" has no instance in the val split' in res.get("error", {}).get("message", ""),
        json.dumps(res.get("error")),
    )


_case_4()


# --- autolabel ----------------------------------------------------------------------
print("--- autolabel")


def _case_5() -> None:
    from xorgate_forge.contracts import LabelsFile

    out = OUT_ROOT / "assist"
    env = {"XORGATE_SECRET_FIXTURE_TOKEN": "hunter2-golden-secret"}
    r = xmf(
        [
            "run",
            str(FIXTURES / "specs" / "autolabel.json"),
            "--dataset",
            str(DS),
            "--models-dir",
            str(FIXTURES / "models"),
            "--out",
            str(out),
            "--progress",
            "json",
        ],
        env=env,
    )
    job = out / "autolabel"
    res = read_json(job / "result.json") if (job / "result.json").exists() else {}
    check(
        "autolabel over fixtures/ds with the pulled yolov8n exits 0",
        r.returncode == 0 and res.get("status") == "succeeded",
        last_line(r.stderr),
    )
    print(f"     (autolabel wall time: {r.seconds:.1f}s)")  # type: ignore[attr-defined]
    try:
        labels = LabelsFile.model_validate(read_json(job / "labels.json"))
        check(
            "labels.json validates and labels every image with car boxes",
            labels.summary.images == 63
            and labels.summary.perClass.get("car", 0) > 50
            and labels.model.name == "yolov8n"
            and labels.model.version == 1,
            json.dumps(labels.summary.to_json_dict()),
        )
        check(
            "boxes carry the model class, the mapped class and absolute xyxy",
            all(
                b.modelCls in ("car", "truck", "bus") and b.cls in ("car", "truck") and b.xyxy[2] > b.xyxy[0]
                for im in labels.images
                for b in im.boxes
            ),
        )
        check(
            "result.labels mirrors the summary and names the file",
            res.get("labels", {}).get("file") == "labels.json"
            and res["labels"]["boxes"] == labels.summary.boxes
            and res.get("artifacts", {}).get("labels") == "labels.json",
        )
        cars = sum(1 for im in labels.images for b in im.boxes if b.cls == "car")
        check(
            "pre-labels agree with the committed fixture within tolerance (92 car boxes)",
            abs(cars - 92) <= 10,
            f"{cars} car boxes",
        )
    except Exception as err:
        check("labels.json validates", False, f"{type(err).__name__}: {err}")
    ndjson_ok = (
        all(json.loads(ln)["t"] in ("status", "log", "epoch") for ln in r.stdout.strip().splitlines())
        if r.stdout.strip()
        else False
    )
    check("--progress json puts only progress NDJSON on stdout", ndjson_ok, r.stdout[:120])
    leaked = [str(p) for p in job.rglob("*") if p.is_file() and "hunter2-golden-secret" in p.read_text(errors="ignore")]
    check(
        "a secret in the environment reaches no artifact or output line",
        not leaked and "hunter2" not in r.stdout + r.stderr,
        ", ".join(leaked),
    )


_case_5()


if SKIP_TRAIN:
    print(f"--- skipped the training cases (XMF_GOLDEN_SKIP_TRAIN=1); {time.monotonic() - started:.1f}s")
    sys.exit(1 if failed else 0)

# --- golden run: quick.json ---------------------------------------------------------
print("--- train (quick)")
quick_model_dir: Path | None = None


def _case_6() -> None:
    global quick_model_dir
    out = OUT_ROOT / "quick"
    r = xmf(
        ["run", str(FIXTURES / "specs" / "quick.json"), "--dataset", str(DS), "--out", str(out), "--progress", "json"]
    )
    print(f"     (quick run wall time: {r.seconds / 60:.1f} min)")  # type: ignore[attr-defined]
    job = out / "quick"
    res = read_json(job / "result.json") if (job / "result.json").exists() else {}
    check(
        "xmf run quick.json exits 0 with result.json succeeded",
        r.returncode == 0 and res.get("status") == "succeeded",
        last_line(r.stderr),
    )
    check(
        "epochsCompleted 12 and effective hyperparameters (imgsz 320, batch 8)",
        res.get("epochsCompleted") == 12
        and res.get("hyperparameters", {}).get("imgsz") == 320
        and res["hyperparameters"].get("batch") == 8,
        json.dumps(res.get("hyperparameters")),
    )
    model_dir = job / "model" / "detection" / "fixture-vehicles" / "1"
    quick_model_dir = model_dir
    check(
        "model/detection/fixture-vehicles/1/ holds manifest.json and model.onnx",
        (model_dir / "manifest.json").is_file()
        and (model_dir / "model.onnx").is_file()
        and res.get("model", {}).get("dir") == "model/detection/fixture-vehicles/1",
    )
    check(
        "check.ok true with four ok lines",
        res.get("check", {}).get("ok") is True
        and len(res["check"]["lines"]) == 4
        and all("  ok   " in ln for ln in res["check"]["lines"]),
        json.dumps(res.get("check")),
    )
    manifest = read_json(model_dir / "manifest.json") if (model_dir / "manifest.json").exists() else {}
    check(
        "manifest v1: classNames in dataset order, yolov8 postprocess, 320x320, files sha256 matches the file",
        manifest.get("manifestVersion") == 1
        and manifest.get("classNames") == ["car", "truck"]
        and manifest.get("postprocess", {}).get("kind") == "yolov8"
        and manifest.get("input", {}).get("width") == 320
        and manifest.get("files", {}).get("model.onnx", {}).get("sha256") == sha256_of(model_dir / "model.onnx"),
    )
    prov = manifest.get("provenance", {})
    check(
        "manifest.provenance names the forge, the recipe license, the base, the job, the dataset hash and metrics",
        prov.get("trainedBy", "").startswith("xorgate-model-forge@")
        and prov.get("recipeLicense") == "AGPL-3.0"
        and prov.get("base", {}).get("ref") == "yolov8n"
        and prov.get("job") == "quick"
        and len(prov.get("dataset", {}).get("hash", "")) == 64
        and "mAP50" in prov.get("metrics", {})
        and isinstance(prov.get("metrics", {}).get("confidence"), float)
        and prov.get("platform", {}).get("orgId") == "fixture",
        json.dumps(prov)[:200],
    )
    metrics = res.get("metrics", {})
    check(
        "metrics: mAP50, mAP50_95, precision, recall, the operating confidence and two per-class rows",
        all(k in metrics for k in ("mAP50", "mAP50_95", "precision", "recall"))
        and [pc["name"] for pc in metrics.get("perClass", [])] == ["car", "truck"]
        and isinstance(metrics.get("confidence"), float)
        and 0.05 <= metrics["confidence"] <= 0.95
        and metrics["confidence"] == prov.get("metrics", {}).get("confidence"),
        json.dumps(metrics)[:200],
    )
    print(
        f"     (quick metrics: mAP50 {metrics.get('mAP50')}, mAP50-95 {metrics.get('mAP50_95')}, precision {metrics.get('precision')}, recall {metrics.get('recall')} at confidence {metrics.get('confidence')})"
    )
    # 3 epochs at 320 measured mAP50 0.0 on this fixture (Ultralytics' first epochs on a 7-batch epoch); 10 reached 0.48.
    check(
        "the quick model detects something on the val split (mAP50 above 0.1 through the exported ONNX)",
        isinstance(metrics.get("mAP50"), float) and metrics["mAP50"] > 0.1,
        f"mAP50 {metrics.get('mAP50')}",
    )
    ev = res.get("artifacts", {}).get("eval", {})
    check(
        "artifacts: train.log, weights/best.pt, eval metrics, confusion and 12 sample renders",
        res.get("artifacts", {}).get("trainLog") == "train.log"
        and (job / "train.log").is_file()
        and (job / "weights" / "best.pt").is_file()
        and (job / ev.get("metrics", "-")).is_file()
        and (job / ev.get("confusion", "-")).is_file()
        and len(ev.get("samples", [])) == 12
        and all((job / s).is_file() for s in ev["samples"]),
        json.dumps(ev)[:200],
    )
    conf = read_json(job / ev["confusion"]) if ev.get("confusion") and (job / ev["confusion"]).exists() else {}
    check(
        "confusion.json is a 3x3 matrix over car, truck, background",
        conf.get("classes") == ["car", "truck", "background"] and len(conf.get("matrix", [])) == 3,
    )
    prog = [json.loads(ln) for ln in (job / "progress.jsonl").read_text().splitlines()]
    phases = [m["phase"] for m in prog if m["t"] == "status"]
    epochs = [m for m in prog if m["t"] == "epoch"]
    check(
        "progress.jsonl: phases prepare, train, export, check, evaluate, done and twelve epoch lines",
        phases == ["prepare", "train", "export", "check", "evaluate", "done"]
        and [e["epoch"] for e in epochs] == list(range(1, 13))
        and all(e["epochs"] == 12 for e in epochs),
        f"{phases} {[e['epoch'] for e in epochs]}",
    )
    check(
        "epoch lines carry loss and mAP50",
        all(isinstance(e.get("loss"), float) and isinstance(e.get("mAP50"), float) for e in epochs),
    )
    stdout_lines: list[Any] = []
    bad_lines: list[str] = []
    for ln in r.stdout.strip().splitlines():
        try:
            stdout_lines.append(json.loads(ln))
        except json.JSONDecodeError:
            bad_lines.append(ln)
    check(
        "--progress json puts only NDJSON on stdout",
        not bad_lines,
        f"first non-JSON line: {bad_lines[0][:200]!r}" if bad_lines else "",
    )
    check(
        "--progress json streamed the same messages to stdout",
        stdout_lines == prog,
        f"{len(stdout_lines)} vs {len(prog)}",
    )
    newest = max((p for p in job.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
    check("result.json is the last file written", newest.name == "result.json", str(newest.relative_to(job)))
    r2 = xmf(["check", str(model_dir), "--no-fill"])
    check(
        "xmf check on the produced directory: four ok lines, sha256 verified",
        r2.returncode == 0
        and "sha256 verified" in r2.stdout
        and r2.stdout.strip().splitlines() == res["check"]["lines"],
        last_line(r2.stdout),
    )


_case_6()


# --- engine cases: xwe models check, xwe run --models-dir ---------------------------
print("--- engine (npx xwe)")
if NODE and XWE.exists() and quick_model_dir is not None and quick_model_dir.exists():
    r = xwe(["models", "check", str(quick_model_dir)])
    lines = r.stdout.strip().splitlines()
    check(
        "npx xwe models check prints four ok lines",
        r.returncode == 0 and len(lines) == 4 and all("  ok   " in ln for ln in lines),
        last_line(r.stdout) or last_line(r.stderr),
    )
    ours = xmf(["check", str(quick_model_dir), "--no-fill"]).stdout.strip().splitlines()
    agree = (
        len(ours) == 4
        and lines[0] == ours[0]
        and lines[1] == ours[1]
        and lines[3] == ours[3]
        and lines[2].replace("onnxruntime-node", "onnxruntime").split(",")[1:] == ours[2].split(",")[1:]
    )
    check(
        "xmf check and xwe models check agree line for line (the session line names each runtime)",
        agree,
        "\n".join(lines + ours),
    )
    run_out = OUT_ROOT / "xwe-run"
    r = xwe(
        [
            "run",
            str(FIXTURES / "templates" / "fixture-vehicles.json"),
            "--input",
            f"video={FIXTURES / 'video' / 'drive.mp4'}",
            "--models-dir",
            str(quick_model_dir.parent.parent.parent),
            "--out",
            str(run_out),
            "--run-id",
            "golden-forge",
        ],
        env={"XWE_MODELS_DIR": str(OUT_ROOT / "empty-cache")},
    )
    check("npx xwe run with --models-dir <out>/quick/model exits 0", r.returncode == 0, last_line(r.stderr))
    print(f"     (xwe run wall time: {r.seconds:.1f}s)")  # type: ignore[attr-defined]
    result = (
        read_json(run_out / "golden-forge" / "result.json")
        if (run_out / "golden-forge" / "result.json").exists()
        else {}
    )
    detect = next((s for s in result.get("steps", []) if s.get("nodeId") == "n3"), {})
    check(
        "the detection step summary names fixture-vehicles v1",
        "(fixture-vehicles v1," in detect.get("output", "") and "over 50 frame(s)" in detect.get("output", ""),
        detect.get("output", ""),
    )
    check(
        "the engine's log line names the forge's model file",
        "[xwe] model detection/fixture-vehicles v1 <- " in r.stderr
        and "quick/model/detection/fixture-vehicles/1/model.onnx" in r.stderr,
        next((ln for ln in r.stderr.splitlines() if "[xwe] model" in ln), ""),
    )
    check(
        "result: succeeded with 5 ok steps",
        result.get("status") == "succeeded"
        and len(result.get("steps", [])) == 5
        and all(s.get("status") == "ok" for s in result["steps"]),
    )
else:
    check("npx xwe cases (node and node_modules/.bin/xwe present)", False, "run `npm ci`")

# --- the SageMaker host under a fake /opt/ml tree ------------------------------------
print("--- host sagemaker")


def _case_7() -> None:
    sm = OUT_ROOT / "opt-ml"
    (sm / "input" / "data").mkdir(parents=True)
    (sm / "input" / "config").mkdir(parents=True)
    shutil.copytree(DS, sm / "input" / "data" / "dataset")
    shutil.copytree(FIXTURES / "models", sm / "input" / "data" / "models")
    spec = read_json(FIXTURES / "specs" / "quick.json")
    spec["jobId"] = "sm-quick"
    spec["hyperparameters"]["epochs"] = 1
    (sm / "input" / "config" / "job.json").write_text(json.dumps(spec))
    r = xmf(["run", str(sm / "input" / "config" / "job.json"), "--host", "sagemaker", "--sagemaker-root", str(sm)])
    print(f"     (sagemaker 1-epoch run wall time: {r.seconds / 60:.1f} min)")  # type: ignore[attr-defined]
    res = read_json(sm / "model" / "result.json") if (sm / "model" / "result.json").exists() else {}
    check(
        "--host sagemaker writes the output tree into /opt/ml/model with result.json succeeded",
        r.returncode == 0 and res.get("status") == "succeeded" and res.get("jobId") == "sm-quick",
        last_line(r.stderr),
    )
    check(
        "progress.jsonl lands in /opt/ml/checkpoints",
        (sm / "checkpoints" / "progress.jsonl").is_file()
        and json.loads(last_line((sm / "checkpoints" / "progress.jsonl").read_text()))["phase"] == "done",
    )
    local_tree = tree(OUT_ROOT / "quick" / "quick") if (OUT_ROOT / "quick" / "quick").exists() else set()
    sm_tree = tree(sm / "model")
    check(
        "the SageMaker output tree equals the local host's",
        local_tree == sm_tree,
        f"only local {sorted(local_tree - sm_tree)[:5]}, only sagemaker {sorted(sm_tree - local_tree)[:5]}",
    )
    check(
        "the dataset came from the /opt/ml/input/data/dataset channel and the model was checked",
        res.get("dataset", {}).get("images", {}).get("train") == 51 and res.get("check", {}).get("ok") is True,
    )
    # The same through XMF_SAGEMAKER_ROOT and the default spec path (the image's entrypoint shape).
    sm2 = OUT_ROOT / "opt-ml-2"
    shutil.copytree(sm / "input", sm2 / "input")
    spec["jobId"] = "sm-env"
    (sm2 / "input" / "config" / "job.json").write_text(json.dumps(spec))
    r = xmf(["validate", "/opt/ml/input/config/job.json", "--host", "sagemaker"], env={"XMF_SAGEMAKER_ROOT": str(sm2)})
    check(
        "XMF_SAGEMAKER_ROOT relocates /opt/ml for validate as well",
        r.returncode == 0 and "OK: job sm-env can run" in r.stdout,
        last_line(r.stderr),
    )


_case_7()


# --- limits.maxMinutes stops training cleanly ----------------------------------------
print("--- limits")


def _case_8() -> None:
    out = OUT_ROOT / "budget"
    r = xmf(["run", str(FIXTURES / "specs" / "budget.json"), "--dataset", str(DS), "--out", str(out)])
    job = out / "budget"
    res = read_json(job / "result.json") if (job / "result.json").exists() else {}
    prog = (
        [json.loads(ln) for ln in (job / "progress.jsonl").read_text().splitlines()]
        if (job / "progress.jsonl").exists()
        else []
    )
    budget_line = next((m["line"] for m in prog if m["t"] == "log" and m["line"].startswith("time budget:")), "")
    check(
        "maxMinutes 1 stops a 50-epoch job cleanly and still exports, checks and evaluates",
        r.returncode == 0
        and res.get("status") == "succeeded"
        and 0 < res.get("epochsCompleted", 0) < 50
        and res.get("check", {}).get("ok") is True
        and budget_line != "",
        f"epochs {res.get('epochsCompleted')}; {budget_line or last_line(r.stderr)}",
    )
    print(f"     (budget run wall time: {r.seconds / 60:.1f} min, {res.get('epochsCompleted')} epoch(s))")  # type: ignore[attr-defined]
    named = re.search(r"during epoch (\d+)", budget_line)
    epoch_lines = [m["epoch"] for m in prog if m["t"] == "epoch"]
    check(
        "epochsCompleted equals the epoch the budget cut short, and the epoch lines stop there",
        named is not None
        and res.get("epochsCompleted") == int(named.group(1))
        and epoch_lines == list(range(1, int(named.group(1)) + 1)),
        f"epochsCompleted {res.get('epochsCompleted')}, budget line {budget_line!r}, epoch lines {len(epoch_lines)}",
    )


_case_8()


elapsed = time.monotonic() - started
print(
    f"--- {'FAILED' if failed else 'all green'}; suite duration {elapsed / 60:.1f} min ({elapsed:.0f}s); artifacts under {OUT_ROOT}"
)
sys.exit(1 if failed else 0)
