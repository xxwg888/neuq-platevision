#!/usr/bin/env python
"""Comprehensive OCR evaluation -> one rich JSON (overall / by-type / by-province /
plate-instance leakage split). Feeds the metrics-table builder and the report.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

import torch
from torch.utils.data import DataLoader

from plate_course.chars import greedy_decode
from plate_course.dataset import PlateOCRDataset, collate_plate_batch
from plate_course.metrics import recognition_metrics
from plate_course.runtime import build_model_from_checkpoint, get_checkpoint_image_size


def acc_pair(pairs):
    if not pairs:
        return {"plate_accuracy": 0.0, "character_accuracy": 0.0, "samples": 0}
    pa = sum(p == t for p, t in pairs) / len(pairs)
    ca = sum(sum(a == b for a, b in zip(p, t)) / max(len(t), 1) for p, t in pairs) / len(pairs)
    return {"plate_accuracy": round(pa, 4), "character_accuracy": round(ca, 4), "samples": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-checkpoint", default="outputs/models/ocr_best.pt")
    ap.add_argument("--test-manifest", default="data/processed/ocr_5color/manifests/test.jsonl")
    ap.add_argument("--train-manifest", default="data/processed/ocr_5color/manifests/train.jsonl")
    ap.add_argument("--output-json", default="outputs/metrics/ocr_final.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ocr_checkpoint, map_location=dev)
    model = build_model_from_checkpoint(ck, device=dev)
    isz = get_checkpoint_image_size(ck)

    ds = PlateOCRDataset(args.test_manifest, image_size=isz)
    ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, collate_fn=collate_plate_batch)
    P, T, R = [], [], []
    with torch.no_grad():
        for b in ld:
            P += greedy_decode(model(b["images"].to(dev)))
            T += b["labels"]
            R += b["records"]

    overall = recognition_metrics(P, T)
    overall["samples"] = len(T)

    by_type = {}
    g = collections.defaultdict(list)
    for p, t, r in zip(P, T, R):
        g[r.get("plate_type", "unknown")].append((p, t))
    for k, v in sorted(g.items()):
        by_type[k] = acc_pair(v)

    by_province = {}
    gp = collections.defaultdict(list)
    for p, t in zip(P, T):
        gp[t[0]].append((p, t))
    for k, v in sorted(gp.items(), key=lambda kv: -len(kv[1])):
        by_province[k] = acc_pair(v)

    trainpn = set(json.loads(l)["plate_number"] for l in open(args.train_manifest, encoding="utf-8") if l.strip())
    seen = [(p, t) for p, t in zip(P, T) if t in trainpn]
    unseen = [(p, t) for p, t in zip(P, T) if t not in trainpn]
    leakage = {"seen_in_train": acc_pair(seen), "unseen_plates_honest": acc_pair(unseen)}

    out = {
        "model": "CRNN-CTC (province-balanced, 5-colour real data)",
        "checkpoint": str(Path(args.ocr_checkpoint).resolve()),
        "overall": {
            "plate_accuracy": round(overall["plate_accuracy"], 4),
            "character_accuracy": round(overall["character_accuracy"], 4),
            "avg_edit_distance": round(overall["avg_edit_distance"], 4),
            "samples": overall["samples"],
            "fps": round(overall.get("fps", 0), 1) if "fps" in overall else None,
        },
        "by_plate_type": by_type,
        "by_province": by_province,
        "leakage_check": leakage,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print("overall:", out["overall"])
    print("by_type:", {k: v["plate_accuracy"] for k, v in by_type.items()})
    print("leakage:", {k: v["plate_accuracy"] for k, v in leakage.items()})


if __name__ == "__main__":
    main()
