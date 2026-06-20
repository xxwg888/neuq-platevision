#!/usr/bin/env python
"""Plate-colour classification evaluation for the YOLO-pose detector.

Runs the detector on each test scene, matches the highest-IoU detection to the GT
box, and compares the predicted colour class against the ground-truth colour. Emits
per-colour accuracy + a confusion matrix. Directly answers "how many plate colours
can we tell apart, and how well" — the course's bonus criterion.

GT colour key from manifest (plate_type + mode):
    blue / green / white                  -> same
    yellow + single_layer                 -> yellow_single
    yellow + double_layer                 -> yellow_double
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

from infer import YOLODetector  # noqa: E402

CLASSES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]


def gt_color(rec):
    t = rec.get("plate_type", "")
    if t == "yellow":
        return "yellow_double" if "double" in rec.get("mode", "") else "yellow_single"
    return t


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-model", default="outputs/yolo_pose/best.pt")
    ap.add_argument("--test-manifest", default="data/processed/ocr_5color/manifests/test.jsonl")
    ap.add_argument("--output-json", default="outputs/metrics/color_eval.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    det = YOLODetector(args.yolo_model, conf=0.25, iou=0.45, imgsz=640, device=args.device)
    records = [json.loads(l) for l in Path(args.test_manifest).open(encoding="utf-8") if l.strip()]

    confusion = defaultdict(Counter)  # gt -> Counter(pred)
    matched = 0
    for rec in records:
        scene = cv2.imread(rec["scene_path"])
        if scene is None:
            continue
        dets = det.detect(scene)
        best, best_iou = None, 0.0
        for d in dets:
            i = iou(d["box"], rec["box_xywh"])
            if i > best_iou:
                best, best_iou = d, i
        if best is None or best_iou < args.iou_threshold:
            continue
        matched += 1
        confusion[gt_color(rec)][best["plate_type"]] += 1

    per_color = {}
    total_correct = 0
    total = 0
    for gt in CLASSES:
        row = confusion.get(gt, Counter())
        n = sum(row.values())
        c = row.get(gt, 0)
        if n:
            per_color[gt] = {"accuracy": round(c / n, 4), "n": n, "pred_dist": dict(row)}
            total_correct += c
            total += n

    out = {
        "model": str(Path(args.yolo_model).resolve()),
        "matched_plates": matched,
        "overall_color_accuracy": round(total_correct / max(total, 1), 4),
        "per_color": per_color,
        "confusion_matrix": {gt: dict(confusion.get(gt, Counter())) for gt in CLASSES if confusion.get(gt)},
        "note": "colour accuracy on plates correctly localised (IoU>=%.2f)." % args.iou_threshold,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print("overall colour accuracy:", out["overall_color_accuracy"], "on", total, "matched plates")
    for k, v in per_color.items():
        print(f"  {k:14s} acc={v['accuracy']:.4f} n={v['n']}")


if __name__ == "__main__":
    main()
