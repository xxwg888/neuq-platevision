#!/usr/bin/env python
"""Template-matching baseline for plate character recognition (classic DIP method).

This is the method the course brief explicitly suggests: build one template per
character class and match by correlation. It shares the EXACT segmentation pipeline
with the KNN baseline (binarize -> vertical-projection char segmentation), so the
three-way comparison (template / HOG+KNN / deep CRNN) differs only in the classifier.

Templates are learned from training data automatically (mean glyph per class), then
each segmented test character is matched by normalized cross-correlation (NCC); the
highest-correlation class wins. No synthetic data.

Usage:
    python scripts/template_baseline.py \
        --train-manifest data/processed/ocr_5color/manifests/train.jsonl \
        --test-manifest  data/processed/ocr_5color/manifests/test.jsonl \
        --output-json outputs/metrics/template_baseline.json \
        --save-templates outputs/models/template/templates.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from plate_course.metrics import recognition_metrics
# Reuse the IDENTICAL segmentation + IO used by the KNN baseline (fair comparison).
from knn_baseline import (
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    binarize,
    load_plate,
    read_jsonl,
    segment_characters_projection,
    segment_equal_split,
)

TPL_H, TPL_W = 40, 24  # template canvas (same aspect as KNN char window)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--output-json", default="outputs/metrics/template_baseline.json")
    p.add_argument("--save-templates", default="outputs/models/template/templates.npz")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--plate-height", type=int, default=48)
    p.add_argument("--plate-width", type=int, default=160)
    return p.parse_args()


def resized_glyph(seg: np.ndarray) -> np.ndarray:
    """Binary char patch -> fixed-size float32 glyph in [0,1]."""
    r = cv2.resize(seg, (TPL_W, TPL_H), interpolation=cv2.INTER_AREA)
    return r.astype(np.float32) / 255.0


def ncc_vector(glyph: np.ndarray) -> np.ndarray:
    """Flatten to a zero-mean, unit-norm vector so a dot product equals NCC."""
    v = glyph.flatten()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def segment_plate(img_bgr: np.ndarray, n_chars: int | None) -> list[np.ndarray]:
    bw = binarize(img_bgr)
    segs = segment_characters_projection(bw, n_chars)
    if (not segs or (n_chars and len(segs) != n_chars)) and n_chars:
        segs = segment_equal_split(bw, n_chars)
    return segs


def build_templates(records, plate_w, plate_h, max_samples):
    """Average glyph per character class -> one NCC template each."""
    acc = collections.defaultdict(lambda: np.zeros((TPL_H, TPL_W), dtype=np.float64))
    cnt = collections.Counter()
    used = 0
    recs = records[:max_samples] if max_samples else records
    for rec in recs:
        img_path = rec.get("image_path")
        plate_number = rec.get("plate_number", "")
        if not img_path or not plate_number or not Path(img_path).is_file():
            continue
        img = load_plate(img_path, plate_w, plate_h)
        if img is None:
            continue
        segs = segment_plate(img, len(plate_number))
        if len(segs) != len(plate_number):
            continue  # only learn from cleanly-segmented plates
        for seg, ch in zip(segs, plate_number):
            if ch in CHAR_TO_IDX:
                acc[ch] += resized_glyph(seg)
                cnt[ch] += 1
        used += 1

    classes = sorted(acc.keys(), key=lambda c: CHAR_TO_IDX[c])
    mats = np.stack([ncc_vector(acc[c] / cnt[c]) for c in classes], axis=0)  # (C, D)
    return classes, mats.astype(np.float32), dict(cnt), used


def predict_plate(img_bgr, n_chars, classes, mats, plate_w, plate_h) -> str:
    segs = segment_plate(img_bgr, n_chars)
    if not segs:
        return ""
    out = []
    for seg in segs:
        v = ncc_vector(resized_glyph(seg)).astype(np.float32)
        scores = mats @ v
        out.append(classes[int(np.argmax(scores))])
    return "".join(out)


def main():
    args = parse_args()

    print("Building character templates from training data...")
    train_records = read_jsonl(args.train_manifest)
    classes, mats, cnt, used = build_templates(
        train_records, args.plate_width, args.plate_height, args.max_train_samples
    )
    print(f"  Learned {len(classes)} class templates from {used} cleanly-segmented plates")

    if args.save_templates:
        Path(args.save_templates).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_templates, classes=np.array(classes), templates=mats)
        print(f"  Saved templates to {args.save_templates}")

    print("Evaluating on test set...")
    test_records = read_jsonl(args.test_manifest)
    if args.max_test_samples:
        test_records = test_records[: args.max_test_samples]

    predictions, targets, records = [], [], []
    t0 = time.time()
    for rec in test_records:
        img_path = rec.get("image_path")
        plate_number = rec.get("plate_number", "")
        targets.append(plate_number)
        records.append(rec)
        if not img_path or not Path(img_path).is_file():
            predictions.append("")
            continue
        img = load_plate(img_path, args.plate_width, args.plate_height)
        if img is None:
            predictions.append("")
            continue
        n_chars = len(plate_number) if plate_number else None
        predictions.append(predict_plate(img, n_chars, classes, mats, args.plate_width, args.plate_height))
    elapsed = time.time() - t0

    metrics = recognition_metrics(predictions, targets)
    metrics.update({
        "samples": len(test_records),
        "elapsed_ms_per_image": elapsed * 1000 / max(len(test_records), 1),
        "fps": len(test_records) / max(elapsed, 1e-6),
    })

    # Per-type breakdown to mirror the deep-model evaluation.
    by_type = collections.defaultdict(lambda: {"pred": [], "tgt": []})
    for p, t, r in zip(predictions, targets, records):
        by_type[r.get("plate_type", "unknown")]["pred"].append(p)
        by_type[r.get("plate_type", "unknown")]["tgt"].append(t)
    by_plate_type = {}
    for k, v in sorted(by_type.items()):
        m = recognition_metrics(v["pred"], v["tgt"])
        m["samples"] = len(v["tgt"])
        by_plate_type[k] = m

    result = {
        "model": "template_matching_NCC_baseline",
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "metrics": metrics,
        "by_plate_type": by_plate_type,
        "success_examples": [{"target": t, "prediction": p} for p, t in zip(predictions, targets) if p == t][:20],
        "failure_examples": [{"target": t, "prediction": p} for p, t in zip(predictions, targets) if p != t][:20],
        "note": (
            "Classic template matching: mean-glyph template per class, NCC matching on the "
            "SAME projection segmentation as the KNN baseline. Weak on tilt/blur and on "
            "many-stroke province glyphs — this is the motivation for the deep recognizer."
        ),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
