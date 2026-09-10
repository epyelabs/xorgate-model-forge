"""`xmf doctor`: Python, torch and CUDA presence, ONNX Runtime, the recipes
and their licenses, weights baked or cached, the model cache root, so "why
does my job fail" has a one-command answer."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .hosts.local import LocalModelSource, baked_weights_dir, default_cache_dir, default_weights_cache_dir
from .recipes import list_recipes


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _list_models(root: Path) -> list[str]:
    found: list[str] = []
    if not root.is_dir():
        return found
    for family in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in sorted(p for p in family.iterdir() if p.is_dir()):
            versions = sorted(
                (
                    int(v.name)
                    for v in name.iterdir()
                    if v.is_dir() and v.name.isdigit() and (v / "model.onnx").is_file()
                ),
                reverse=True,
            )
            if versions:
                found.append(
                    f"{family.name}/{name.name} v{versions[0]}"
                    + (f" (+{len(versions) - 1})" if len(versions) > 1 else "")
                )
    return found


def doctor_report() -> dict[str, Any]:
    report: dict[str, Any] = {"xmf": __version__, "python": sys.version.split()[0], "platform": platform.platform()}
    torch_info: dict[str, Any] = {"installed": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "cudaVersion": getattr(torch.version, "cuda", None),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "device": os.environ.get("XMF_DEVICE") or ("0" if torch.cuda.is_available() else "cpu"),
        }
    except Exception as err:
        torch_info["error"] = f"{type(err).__name__}: {err}"
    report["torch"] = torch_info
    ort: dict[str, Any] = {"installed": False}
    try:
        import onnxruntime

        ort = {
            "installed": True,
            "version": onnxruntime.__version__,
            "providers": onnxruntime.get_available_providers(),
        }
    except Exception as err:
        ort["error"] = f"{type(err).__name__}: {err}"
    report["onnxruntime"] = ort
    report["ultralytics"] = _version("ultralytics")
    source = LocalModelSource()
    weights_dirs = [baked_weights_dir(), default_weights_cache_dir()]
    recipes = []
    for r in list_recipes():
        bases = {}
        for ref in r.base_refs:
            found = next((str(d / f"{ref}.pt") for d in weights_dirs if (d / f"{ref}.pt").is_file()), None)
            bases[ref] = found
        recipes.append({"id": r.id, "family": r.family, "task": r.task, "license": r.license, "baseWeights": bases})
    report["recipes"] = recipes
    report["weightsDirs"] = [str(d) for d in weights_dirs]
    report["weightsPull"] = os.environ.get("XMF_WEIGHTS_PULL", "1") != "0"
    cache = default_cache_dir()
    report["modelCache"] = {"root": str(cache), "models": _list_models(cache)}
    report["modelSearch"] = source.describe()
    node = shutil.which("node")
    report["node"] = {
        "path": node,
        "xwe": shutil.which("xwe")
        or (str(Path("node_modules/.bin/xwe")) if Path("node_modules/.bin/xwe").exists() else None),
    }
    report["sagemakerRoot"] = os.environ.get("XMF_SAGEMAKER_ROOT") or "/opt/ml"
    return report


def render_doctor(report: dict[str, Any]) -> str:
    lines = [f"xmf {report['xmf']}  python {report['python']}  {report['platform']}"]
    t = report["torch"]
    if t.get("installed"):
        accel = "cuda " + str(t.get("cudaVersion")) if t.get("cuda") else ("mps" if t.get("mps") else "cpu only")
        lines.append(f"torch        ok   {t['version']} ({accel}); training device {t['device']}")
    else:
        lines.append(f"torch        missing   install xorgate-forge[yolov8] ({t.get('error', 'not installed')})")
    o = report["onnxruntime"]
    if o.get("installed"):
        lines.append(f"onnxruntime  ok   {o['version']} ({', '.join(o['providers'])})")
    else:
        lines.append("onnxruntime  missing   install xorgate-forge[assist] (check, evaluate and autolabel need it)")
    lines.append(
        "ultralytics  "
        + ("ok   " + report["ultralytics"] if report["ultralytics"] else "missing   install xorgate-forge[yolov8]")
    )
    for r in report["recipes"]:
        bases = ", ".join(
            f"{ref} {'baked' if p and p.startswith(report['weightsDirs'][0]) else ('cached' if p else 'absent')}"
            for ref, p in r["baseWeights"].items()
        )
        lines.append(
            f"recipe       {r['id']:<8} {r['family']}, {r['task']}, license {r['license']}; base weights: {bases}"
        )
    lines.append(
        f"weights      {', '.join(report['weightsDirs'])}; downloads "
        + ("on" if report["weightsPull"] else "off (XMF_WEIGHTS_PULL=0)")
    )
    mc = report["modelCache"]
    lines.append(
        f"model cache  {mc['root']}: "
        + (", ".join(mc["models"]) if mc["models"] else "empty (xwe models pull detection/yolov8n fills it)")
    )
    n = report["node"]
    lines.append(f"node         {n['path'] or 'absent'}; xwe {n['xwe'] or 'absent (only the golden suite needs it)'}")
    lines.append(f"sagemaker    root {report['sagemakerRoot']}")
    return "\n".join(lines)
