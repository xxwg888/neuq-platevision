#!/usr/bin/env python
"""Targeted CRPD_all extractor that guarantees coverage of rare plate types.

Random sampling would barely capture yellow_double (160 total) / white (433 total),
so this scans all labels, takes ALL of the rare types plus a cap of the common ones,
respects CRPD's own train/val/test split (from the path), reads each scene image once,
crops every selected plate, and writes a unified OCR manifest.

Plate type from CRPD label field 9: 0 blue, 1 yellow_single, 2 yellow_double, 3 white.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

TYPE_MAP = {
    "0": ("blue", "single_layer"),
    "1": ("yellow", "single_layer"),
    "2": ("yellow", "double_layer"),
    "3": ("white", "single_layer"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="/var/tmp/plate_data_cxj/archives/CRPD_all.zip")
    p.add_argument("--output-dir", default="/var/tmp/plate_data_cxj/processed/crpd_targeted")
    p.add_argument("--cap-blue", type=int, default=3000)
    p.add_argument("--cap-yellow-single", type=int, default=2500)
    p.add_argument("--crop-height", type=int, default=96)
    p.add_argument("--crop-width", type=int, default=320)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_plate(img, corners, width, height):
    src = order_points(np.asarray(corners, dtype=np.float32))
    dst = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (width, height))


def split_of(path: str) -> str:
    low = path.lower()
    for s in ("train", "val", "test"):
        if f"/{s}/" in low:
            return s
    return "train"


def parse_line(line: str):
    raw = line.strip()
    parts = raw.split(",") if "," in raw else raw.split()
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 9:
        return None
    nums = [int(float(v)) for v in parts[:8]]
    corners = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
    ptype, mode = TYPE_MAP.get(parts[8], ("unknown", "unknown"))
    content = "".join(parts[9:]) if len(parts) > 9 else ""
    return corners, ptype, mode, content


def label_for_image(img_name: str, name_set: set) -> str | None:
    if "/images/" in img_name:
        cand = img_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if cand in name_set:
            return cand
    sib = img_name.rsplit(".", 1)[0] + ".txt"
    return sib if sib in name_set else None


def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.output_dir)
    crop_dir = out / "crops"
    scene_dir = out / "images"
    man_dir = out / "manifests"
    for d in (crop_dir, scene_dir, man_dir):
        d.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.archive)
    names = zf.namelist()
    name_set = set(names)
    images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Pass 1: index every labeled plate (cheap: labels only).
    plates = []  # (img_name, split, line_idx, corners, ptype, mode, content)
    for img_name in images:
        lbl = label_for_image(img_name, name_set)
        if not lbl:
            continue
        sp = split_of(img_name)
        try:
            txt = zf.read(lbl).decode("utf-8", "ignore")
        except Exception:
            continue
        for li, line in enumerate(txt.splitlines()):
            parsed = parse_line(line)
            if not parsed:
                continue
            corners, ptype, mode, content = parsed
            if not content:
                continue
            plates.append((img_name, sp, li, corners, ptype, mode, content))

    # Select: ALL rare types, cap common types (per split-proportional).
    buckets = collections.defaultdict(list)
    for rec in plates:
        key = (rec[5] == "double_layer", rec[4])  # (is_double, ptype)
        buckets[key].append(rec)

    selected = []
    for key, recs in buckets.items():
        is_double, ptype = key
        random.shuffle(recs)
        if is_double or ptype == "white":
            selected.extend(recs)              # take ALL rare
        elif ptype == "blue":
            selected.extend(recs[: args.cap_blue])
        elif ptype == "yellow":
            selected.extend(recs[: args.cap_yellow_single])
        else:
            selected.extend(recs[:500])

    # Group selected plates by image so each scene is read once.
    by_image = collections.defaultdict(list)
    for rec in selected:
        by_image[rec[0]].append(rec)

    records = []
    rec_id = 0
    for img_name, recs in by_image.items():
        data = np.frombuffer(zf.read(img_name), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        scene_saved = None
        for (_, sp, li, corners, ptype, mode, content) in recs:
            crop = crop_plate(img, corners, args.crop_width, args.crop_height)
            crop_path = crop_dir / f"{rec_id:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            if scene_saved is None:
                scene_saved = scene_dir / f"{Path(img_name).stem}_{rec_id:06d}.jpg"
                cv2.imwrite(str(scene_saved), img)
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            records.append({
                "id": f"crpd_{rec_id:06d}",
                "image_path": str(crop_path.resolve()),
                "scene_path": str(scene_saved.resolve()),
                "plate_number": content,
                "plate_type": ptype,
                "mode": mode,
                "source": "CRPD",
                "split": sp,
                "corners": corners,
                "box_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "archive_member": img_name,
            })
            rec_id += 1

    # Write split manifests honoring CRPD's official train/val/test.
    splits = collections.defaultdict(list)
    for r in records:
        splits[r["split"]].append(r)
    for sp in ("train", "val", "test"):
        with (man_dir / f"{sp}.jsonl").open("w", encoding="utf-8") as f:
            for r in splits.get(sp, []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (man_dir / "all.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    type_cnt = collections.Counter(r["plate_type"] for r in records)
    mode_cnt = collections.Counter(r["mode"] for r in records)
    summary = {
        "source": "CRPD_all targeted",
        "total": len(records),
        "by_split": {k: len(v) for k, v in splits.items()},
        "plate_type_counts": dict(type_cnt),
        "mode_counts": dict(mode_cnt),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
