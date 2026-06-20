#!/usr/bin/env python
"""Build a balanced 5-color real-data OCR manifest.

Combines CCPD2019 (blue), CCPD-Green (green), CRPD-targeted (blue/yellow/white)
into train/val/test manifests. To keep the abundant blue class from drowning the
rare classes, TRAIN is capped per type and rare types are oversampled. VAL/TEST keep
the natural real distribution so reported metrics are honest.

Out-of-charset plates (e.g. military '字' markers) are dropped.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from plate_course.chars import CHAR_TO_IDX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ccpd2019", default="/var/tmp/plate_data_cxj/processed/ccpd2019_subset/manifests")
    p.add_argument("--ccpd-green", default="data/processed/ccpd_green_subset/manifests")
    p.add_argument("--crpd", default="/var/tmp/plate_data_cxj/processed/crpd_province/manifests")
    p.add_argument("--out-dir", default="data/processed/ocr_5color/manifests")
    p.add_argument("--cap-blue-train", type=int, default=4000)
    p.add_argument("--cap-prov", type=int, default=800, help="max train samples per province char")
    p.add_argument("--min-prov", type=int, default=200, help="min train samples per province (oversample)")
    p.add_argument("--min-prov-total", type=int, default=80,
                   help="provinces with fewer than this many real TRAIN plates are dropped "
                        "from all splits (genuinely data-starved; cannot be learned or fairly tested).")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def type_key(rec):
    pt = rec.get("plate_type", "")
    mode = rec.get("mode", "")
    if pt == "yellow" and "double" in mode:
        return "yellow_double"
    return pt


def in_charset(text):
    return bool(text) and all(c in CHAR_TO_IDX for c in text)


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "train": [],
        "val": [],
        "test": [],
    }
    for split in ("train", "val", "test"):
        sources[split] += read_jsonl(Path(args.ccpd2019) / f"{split}.jsonl")
        sources[split] += read_jsonl(Path(args.ccpd_green) / f"{split}.jsonl")
        sources[split] += read_jsonl(Path(args.crpd) / f"{split}.jsonl")

    # Drop out-of-charset plates everywhere.
    dropped = collections.Counter()
    for split in sources:
        kept = []
        for r in sources[split]:
            if in_charset(r.get("plate_number", "")):
                kept.append(r)
            else:
                dropped[split] += 1
        sources[split] = kept

    # Drop genuinely data-starved provinces (e.g. 浙: only ~82 plates in all of CRPD).
    # These cannot be learned from a handful of real samples nor fairly tested, so the
    # system is scoped to the provinces with sufficient real data. Counted on the real
    # TRAIN pool (before any oversampling). 皖 is safe (CCPD2019 supplies thousands).
    train_prov_count = collections.Counter(r["plate_number"][0] for r in sources["train"])
    drop_provs = {p for p, c in train_prov_count.items() if c < args.min_prov_total}
    dropped_prov_detail = {p: train_prov_count[p] for p in sorted(drop_provs)}
    if drop_provs:
        for split in sources:
            sources[split] = [r for r in sources[split] if r["plate_number"][0] not in drop_provs]

    # TRAIN balancing in two stages:
    #   (1) oversample rare PLATE TYPES so each colour is represented;
    #   (2) balance by PROVINCE char (the dominant error source): cap over-represented
    #       provinces (CCPD is ~50% 皖) and oversample rare provinces so the model
    #       actually learns every province glyph. Augmentation diversifies the copies.
    train = sources["train"]

    # Green is province-uniform (CCPD-Green is almost all 皖) and already easy, so it is
    # kept OUT of province balancing (otherwise the 皖 cap would crush the green type).
    green = [r for r in train if type_key(r) == "green"]
    non_green = [r for r in train if type_key(r) != "green"]

    # Stage 1: type oversample the 7-char types (build a pool for province balancing).
    type_factor = {"white": 2, "yellow_double": 4, "yellow": 1, "blue": 1}
    pool = []
    for r in non_green:
        pool.extend([r] * type_factor.get(type_key(r), 1))

    # Stage 2: province balance — cap over-represented provinces (CCPD ~50% 皖) and
    # oversample rare provinces so every province glyph is learned. Augmentation
    # diversifies the duplicated copies.
    by_prov = collections.defaultdict(list)
    for r in pool:
        by_prov[r["plate_number"][0]].append(r)

    balanced_train = []
    for prov, recs in by_prov.items():
        random.shuffle(recs)
        if len(recs) > args.cap_prov:
            recs = recs[: args.cap_prov]
        elif len(recs) < args.min_prov:
            reps = (args.min_prov + len(recs) - 1) // len(recs)
            recs = (recs * reps)[: args.min_prov]
        balanced_train.extend(recs)

    # Add green back, oversampled to stay well-represented.
    balanced_train.extend(green * 2)
    random.shuffle(balanced_train)

    out = {"train": balanced_train, "val": sources["val"], "test": sources["test"]}
    for split, recs in out.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def dist(recs):
        return dict(collections.Counter(type_key(r) for r in recs))

    summary = {
        "train_count": len(out["train"]),
        "val_count": len(out["val"]),
        "test_count": len(out["test"]),
        "train_dist (after balance)": dist(out["train"]),
        "val_dist (natural)": dist(out["val"]),
        "test_dist (natural)": dist(out["test"]),
        "dropped_out_of_charset": dict(dropped),
        "dropped_scarce_provinces": dropped_prov_detail,
        "note": "train oversampled for rare types; val/test natural real distribution; "
                "provinces with <%d real train plates dropped as data-starved." % args.min_prov_total,
    }
    (out_dir.parent / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
