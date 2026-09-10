"""The `xmf` command. Exit codes as the engine's: 0 success, 1 the job or
the validation failed (`result.json` written for runs), 2 usage or
configuration (nothing ran). `--log-level` gates stderr; stdout carries
results and the progress stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import __version__
from ..check import check_model_dir
from ..contracts import AutolabelJobSpec, TrainJobSpec
from ..dataset import assign_splits, compute_stats, dataset_hash, load_dataset, validate_dataset, write_dataset
from ..dataset.formats import export_coco, export_yolo, import_coco, import_voc, import_yolo
from ..doctor import doctor_report, render_doctor
from ..hosts import LocalHost, LocalModelSource, SageMakerHost, sagemaker_paths
from ..hosts.local import baked_weights_dir, default_weights_cache_dir
from ..recipes import get_recipe, list_recipes
from ..runner import UsageError, load_spec, run_job
from ..seams import Host
from ..util import LOG, configure_logging, read_json

USAGE = """usage:
  xmf doctor
  xmf recipes [--json]
  xmf validate <job.json> --dataset <dir> [--models-dir <dir>]...
  xmf run <job.json> --dataset <dir> --out <dir> [--models-dir <dir>]... [--progress json] [--host local|sagemaker]
  xmf train <job.json> --dataset <dir> --out <dir> [...]        (run, task train only)
  xmf dataset validate <dir> [--min-val-instances n]
  xmf dataset import <src> --from coco|yolo|voc --out <dir> [--images <dir>] [--classes a,b] [--split train]
  xmf dataset export <dir> --to coco|yolo --out <dir>
  xmf dataset split <dir> [--train 0.8] [--val 0.2] [--test 0] [--seed 0] [--keep <splits.json>]
  xmf dataset stats <dir> [--json]
  xmf check <model-dir> [--no-fill]
  xmf weights pull [<ref>...] [--dir <dir>]
  every command: --log-level silent|error|info|debug"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # exit 2 with the usage block, like xwe
        sys.stderr.write(f"xmf: {message}\n{USAGE}\n")
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="xmf", add_help=True, description="xorgate-model-forge", usage=USAGE)
    p.add_argument("--version", action="version", version=f"xmf {__version__}")
    p.add_argument("--log-level", choices=["silent", "error", "info", "debug"], default="info")
    # Accepted before or after the subcommand, as on `xwe`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", choices=["silent", "error", "info", "debug"], default=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)
    sub_parents: dict[str, Any] = {"parents": [common]}

    sub.add_parser("doctor", help="what this machine can run", **sub_parents)
    r = sub.add_parser("recipes", help="recipes and their licenses", **sub_parents)
    r.add_argument("--json", action="store_true")

    def run_args(sp: argparse.ArgumentParser, *, need_out: bool) -> None:
        sp.add_argument("spec", type=Path, help="job.json")
        sp.add_argument("--dataset", type=Path, help="a dataset v1 directory (local host)")
        if need_out:
            sp.add_argument("--out", type=Path, help="output root; artifacts land in <out>/<jobId>/ (local host)")
        sp.add_argument("--models-dir", type=Path, action="append", default=[], help="a registry root (repeatable)")
        sp.add_argument("--host", choices=["local", "sagemaker"], default="local")
        sp.add_argument("--sagemaker-root", type=Path, help="/opt/ml for the SageMaker host (XMF_SAGEMAKER_ROOT)")
        sp.add_argument("--secrets", type=Path, help="JSON file of logical secret names to values")
        if need_out:
            sp.add_argument("--progress", choices=["none", "json"], default="none", help="NDJSON progress on stdout")

    v = sub.add_parser("validate", help="what a job needs; exit 2 on a bad spec", **sub_parents)
    run_args(v, need_out=False)
    run = sub.add_parser("run", help="run a job", **sub_parents)
    run_args(run, need_out=True)
    train = sub.add_parser("train", help="run a train job", **sub_parents)
    run_args(train, need_out=True)

    ds = sub.add_parser("dataset", help="dataset tools", **sub_parents)
    dsub = ds.add_subparsers(dest="dataset_command", metavar="<subcommand>", parser_class=_Parser)
    dv = dsub.add_parser("validate", **sub_parents)
    dv.add_argument("dir", type=Path)
    dv.add_argument("--min-val-instances", type=int, default=5)
    dv.add_argument("--no-val", action="store_true", help="do not require a val split (an unlabeled import)")
    di = dsub.add_parser("import", **sub_parents)
    di.add_argument("src", type=Path, help="COCO json, a YOLO layout directory, or a VOC directory")
    di.add_argument("--from", dest="fmt", choices=["coco", "yolo", "voc"], required=True)
    di.add_argument("--out", type=Path, required=True)
    di.add_argument("--images", type=Path, help="coco: the images directory (default: next to the json)")
    di.add_argument("--classes", help="yolo, voc: comma-separated class names in id order")
    di.add_argument("--split", choices=["train", "val", "test"], default="train", help="split for images without one")
    de = dsub.add_parser("export", **sub_parents)
    de.add_argument("dir", type=Path)
    de.add_argument("--to", dest="fmt", choices=["coco", "yolo"], required=True)
    de.add_argument("--out", type=Path, required=True)
    dsp = dsub.add_parser("split", **sub_parents)
    dsp.add_argument("dir", type=Path)
    dsp.add_argument("--train", type=float, default=0.8)
    dsp.add_argument("--val", type=float, default=0.2)
    dsp.add_argument("--test", type=float, default=0.0)
    dsp.add_argument("--seed", type=int, default=0)
    dsp.add_argument("--keep", type=Path, help="a previous splits.json whose assignments stay")
    dst = dsub.add_parser("stats", **sub_parents)
    dst.add_argument("dir", type=Path)
    dst.add_argument("--json", action="store_true")

    c = sub.add_parser("check", help="the four-line model directory check", **sub_parents)
    c.add_argument("dir", type=Path)
    c.add_argument("--no-fill", action="store_true", help="do not write files.sha256 into manifest.json")

    w = sub.add_parser("weights", help="base weights", **sub_parents)
    wsub = w.add_subparsers(dest="weights_command", metavar="<subcommand>", parser_class=_Parser)
    wp = wsub.add_parser("pull", **sub_parents)
    wp.add_argument("refs", nargs="*", help="yolov8n, yolov8s ... (default: every recipe base)")
    wp.add_argument("--dir", type=Path, help="target directory (default: the forge weights cache)")
    return p


# --- helpers ----------------------------------------------------------------------


def _bind_host(args: argparse.Namespace, job_id: str, *, need_out: bool) -> Host:
    if args.host == "sagemaker":
        return SageMakerHost(
            job_id=job_id, root=args.sagemaker_root, progress_stdout=getattr(args, "progress", "none") == "json"
        ).bind()
    if args.dataset is None:
        raise UsageError("--dataset <dir> is required on the local host")
    if need_out and args.out is None:
        raise UsageError("--out <dir> is required on the local host")
    return LocalHost(
        dataset_dir=args.dataset,
        out_dir=args.out if need_out else Path("."),
        job_id=job_id,
        models_dirs=args.models_dir,
        progress_stdout=getattr(args, "progress", "none") == "json",
        secrets_file=args.secrets,
    ).bind()


def _spec_path(args: argparse.Namespace) -> Path:
    if args.host == "sagemaker" and not Path(args.spec).is_file():
        return sagemaker_paths(args.sagemaker_root).job_spec
    return Path(args.spec)


def cmd_run(args: argparse.Namespace, *, train_only: bool) -> int:
    spec_path = _spec_path(args)
    spec = load_spec(spec_path)
    if train_only and not isinstance(spec, TrainJobSpec):
        raise UsageError(f'{spec_path}: `xmf train` runs task "train" jobs; this one is "{spec.task}" (use `xmf run`)')
    if args.host == "sagemaker":
        host = SageMakerHost(
            job_id=spec.jobId, root=args.sagemaker_root, progress_stdout=args.progress == "json"
        ).bind()
    else:
        host = _bind_host(args, spec.jobId, need_out=True)
    LOG.info(f"job {spec.jobId} ({spec.task}, {spec.recipe}) on the {host.name} host -> {host.artifacts.dir()}")
    outcome = run_job(spec, host, spec_path=spec_path)
    return outcome.exit_code


def cmd_validate(args: argparse.Namespace) -> int:
    spec_path = _spec_path(args)
    spec = load_spec(spec_path)
    problems: list[str] = []
    lines: list[str] = []
    try:
        recipe = get_recipe(spec.recipe)
    except KeyError as err:
        raise UsageError(str(err)) from err
    lines.append(f"task {spec.task}, recipe {recipe.id} ({recipe.license})")
    if args.host == "sagemaker":
        dataset_dir = sagemaker_paths(args.sagemaker_root).dataset
        models = LocalModelSource(roots=[sagemaker_paths(args.sagemaker_root).models])
    else:
        if args.dataset is None:
            raise UsageError("--dataset <dir> is required")
        dataset_dir = args.dataset
        models = LocalModelSource(roots=args.models_dir)
    report = validate_dataset(dataset_dir, open_images=False, require_val=isinstance(spec, TrainJobSpec))
    if report.ok:
        lines.append(f"dataset {dataset_dir}: {report.summary()}")
    else:
        problems += [f"dataset: {e}" for e in report.errors]
    for w in report.warnings:
        lines.append(f"warning: {w}")
    if isinstance(spec, TrainJobSpec):
        hp = recipe.resolve_hyperparameters(spec)
        lines.append(
            f"output {spec.output.family}/{spec.output.name} v{spec.output.version}, mode {spec.mode}: "
            + ", ".join(f"{k} {v}" for k, v in hp.items())
        )
        if spec.base.kind == "pretrained":
            assert spec.base.ref
            if spec.base.ref not in recipe.base_refs:
                problems.append(f'base "{spec.base.ref}" is not one of {list(recipe.base_refs)}')
            else:
                found = next(
                    (d / f"{spec.base.ref}.pt" for d in models.weights_dirs if (d / f"{spec.base.ref}.pt").is_file()),
                    None,
                )
                lines.append(
                    f"base {spec.base.ref}: "
                    + (str(found) if found else ("will download" if models.allow_download else "absent, downloads off"))
                )
                if found is None and not models.allow_download:
                    problems.append(f"base weights {spec.base.ref} absent and downloads are off")
        else:
            assert spec.base.path
            p = Path(spec.base.path)
            if not p.is_absolute():
                p = spec_path.parent / p
            (lines if p.is_file() else problems).append(f"base weights {p}" + ("" if p.is_file() else " not found"))
        lines.append(f"limits: maxMinutes {spec.limits.maxMinutes}")
    else:
        assert isinstance(spec, AutolabelJobSpec)
        try:
            m = models.fetch(spec.model.family, spec.model.name)
            lines.append(f"model {m.family}/{m.name} v{m.version} <- {m.dir} ({len(m.manifest.classNames)} classes)")
        except (OSError, ValueError, ValidationError) as err:
            problems.append(str(err))
        missing = sorted(set(spec.classMap.values()) - set(report.classes))
        if missing:
            problems.append(f"classMap names class(es) the dataset lacks: {missing}")
    for line in lines:
        print(line)
    if problems:
        for pr in problems:
            print(f"problem: {pr}", file=sys.stderr)
        return 1
    print(f"OK: job {spec.jobId} can run")
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    recipes = list_recipes()
    if args.json:
        print(json.dumps([r.describe() for r in recipes], indent=2))
        return 0
    print(f"{'id':<10}{'family':<14}{'task':<18}{'license':<12}base weights")
    for r in recipes:
        print(f"{r.id:<10}{r.family:<14}{r.task:<18}{r.license:<12}{', '.join(r.base_refs)}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    cmd = args.dataset_command
    if cmd == "validate":
        report = validate_dataset(args.dir, min_val_instances=args.min_val_instances, require_val=not args.no_val)
        for w in report.warnings:
            print(f"warning: {w}")
        for e in report.errors:
            print(f"error: {e}", file=sys.stderr)
        if report.ok:
            print(f"OK: {report.summary()}")
            print(f"hash {dataset_hash(args.dir)}")
            return 0
        print(f"FAIL: {len(report.errors)} error(s) in {args.dir}", file=sys.stderr)
        return 1
    if cmd == "import":
        classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
        if args.fmt == "coco":
            rep = import_coco(args.src, args.out, images_dir=args.images, split=args.split)
        elif args.fmt == "yolo":
            rep = import_yolo(args.src, args.out, classes=classes)
        else:
            rep = import_voc(args.src, args.out, classes=classes, split=args.split)
        print(f"imported {args.fmt} -> {args.out}: {rep.render()}")
        return 0
    if cmd == "export":
        ds = load_dataset(args.dir, require_splits=False)
        target = export_coco(ds, args.out) if args.fmt == "coco" else export_yolo(ds, args.out)
        print(f"exported {args.fmt} -> {target}")
        return 0
    if cmd == "split":
        ds = load_dataset(args.dir, require_splits=False)
        keep = read_json(args.keep) if args.keep else None
        ds.splits = assign_splits(ds.coco, train=args.train, val=args.val, test=args.test, seed=args.seed, keep=keep)
        write_dataset(ds)
        c = ds.header.splits
        print(
            f"split {len(ds.coco.images)} images by group (seed {args.seed}): "
            f"train {c.train}, val {c.val}, test {c.test}"
        )
        return 0
    if cmd == "stats":
        ds = load_dataset(args.dir, require_splits=False)
        stats = compute_stats(ds)
        print(json.dumps(stats.to_json(), indent=2) if args.json else stats.render())
        return 0
    raise UsageError("xmf dataset: validate | import | export | split | stats")


def cmd_check(args: argparse.Namespace) -> int:
    report = check_model_dir(args.dir, fill=not args.no_fill)
    for line in report.rendered():
        print(line)
    return 0 if report.ok else 1


def cmd_weights(args: argparse.Namespace) -> int:
    if args.weights_command != "pull":
        raise UsageError("xmf weights: pull")
    target_dir = args.dir or default_weights_cache_dir()
    refs = args.refs or [ref for r in list_recipes() for ref in r.base_refs]
    source = LocalModelSource(weights_dirs=[baked_weights_dir(), target_dir], allow_download=True)
    for ref in refs:
        path = source.fetch_weights(ref)
        print(f"{ref}: {path} ({path.stat().st_size} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "doctor":
            print(render_doctor(doctor_report()))
            return 0
        if args.command == "recipes":
            return cmd_recipes(args)
        if args.command == "validate":
            return cmd_validate(args)
        if args.command in ("run", "train"):
            return cmd_run(args, train_only=args.command == "train")
        if args.command == "dataset":
            return cmd_dataset(args)
        if args.command == "check":
            return cmd_check(args)
        if args.command == "weights":
            return cmd_weights(args)
        parser.print_usage(sys.stderr)
        return 2
    except UsageError as err:
        print(f"xmf: {err}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, ValidationError) as err:
        print(f"xmf: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
