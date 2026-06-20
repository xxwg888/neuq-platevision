#!/usr/bin/env python
"""Generate axis-aligned bbox crops that MATCH the inference-time crop style.

The perspective-warped GT crops used for the first OCR experiments do not match
what infer.py produces (it crops the YOLO axis-aligned box). This script crops
each scene image by its `box_xywh` (with padding) so OCR training data matches
inference, then writes a new manifest pointing at the bbox crops.

Usage:
    python scripts/make_bbox_crops.py \
        --manifest data/processed/ccpd_green_subset/manifests/train.jsonl \
        --out-crop-dir /tmp/plate_data_cxj/bbox_crops/ccpd_green/train \
        --out-manifest /tmp/plate_data_cxj/bbox_real/ccpd_green/train.jsonl \
        --pad 0.08
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-crop-dir", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--pad", type=float, default=0.08, help="Padding ratio around box.")
    return p.parse_args()


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    args = parse_args()
    crop_dir = Path(args.out_crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.manifest)
    written = 0
    skipped = 0
    out_records = []

    for rec in records:
        scene = rec.get("scene_path")
        box = rec.get("box_xywh")
        if not scene or not Path(scene).is_file() or not box:
            skipped += 1
            continue
        img = cv2.imread(scene)
        if img is None:
            skipped += 1
            continue
        ih, iw = img.shape[:2]
        x, y, w, h = box
        px = w * args.pad
        py = h * args.pad
        x0 = max(0, int(x - px))
        y0 = max(0, int(y - py))
        x1 = min(iw, int(x + w + px))
        y1 = min(ih, int(y + h + py))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            skipped += 1
            continue
        rec_id = rec.get("id", f"rec_{written:06d}")
        crop_path = crop_dir / f"{rec_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        new_rec = dict(rec)
        new_rec["image_path"] = str(crop_path.resolve())
        new_rec["crop_style"] = "bbox_pad"
        out_records.append(new_rec)
        written += 1

    with out_manifest.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{args.manifest}: written={written} skipped={skipped} -> {out_manifest}")


if __name__ == "__main__":
    main()
