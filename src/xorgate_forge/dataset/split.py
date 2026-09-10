"""`xmf dataset split`: assign splits by group, deterministically under a
seed, keeping a previous `splits.json` where asked (design 3b, 7)."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from ..contracts import CocoFile, Split, Splits


def _rank(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(
    coco: CocoFile,
    *,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 0,
    keep: Splits | None = None,
) -> Splits:
    """Groups (an image's `group`, else its own id) are ordered by a seeded
    hash and dealt to the splits by cumulative fraction, largest groups
    first, so the same inputs always produce the same assignment. Images
    named in `keep` stay where they were; the fractions apply to the rest."""
    total = train + val + test
    if total <= 0:
        raise ValueError("split fractions must sum to a positive number")
    fr_train, fr_val = train / total, val / total
    keep = keep or {}
    out: Splits = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for img in coco.images:
        if img.id in keep:
            out[img.id] = keep[img.id]
            continue
        groups[img.group or img.id].append(img.id)
    if not groups:
        return out
    # Deterministic order: seeded hash of the group key.
    ordered = sorted(groups.items(), key=lambda kv: (_rank(kv[0], seed), kv[0]))
    remaining = sum(len(ids) for _, ids in ordered)
    quota = {"train": fr_train * remaining, "val": fr_val * remaining}
    quota["test"] = remaining - quota["train"] - quota["val"]
    filled = {"train": 0.0, "val": 0.0, "test": 0.0}
    for _, ids in ordered:
        # The split with the largest unfilled share takes this group.
        order: list[Split] = ["train", "val", "test"]
        target = max(order, key=lambda s: (quota[s] - filled[s], -order.index(s)))
        if quota[target] - filled[target] <= 0:
            target = "train"
        for image_id in ids:
            out[image_id] = target
        filled[target] += len(ids)
    return out
