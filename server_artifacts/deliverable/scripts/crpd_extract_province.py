#!/usr/bin/env python
"""Province-aware CRPD_all extractor.

CRPD's official train/val/test split is region-stratified: several provinces appear
ONLY in val/test (e.g. 桂 has 862/865 plates in test, ~0 in train). Inheriting that
split starves the recognizer of those province glyphs, so it can never learn them.

This extractor instead pools ALL CRPD splits, caps each province to keep 川 (11.8k)
from dominating, always keeps the rare *types* (white, yellow_double), then RE-SPLITS
image-disjoint 70/15/15 *within each province* so every province is represented in
train, val, and test. No image appears in two splits (no leakage). Real data only.

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

PROVINCES = set("京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="/var/tmp/plate_data_cxj/archives/CRPD_all.zip")
    p.add_argument("--output-dir", default="/var/tmp/plate_data_cxj/processed/crpd_province")
    p.add_argument("--cap-prov", type=int, default=400, help="max plates kept per province")
    p.add_argument("--crop-height", type=int, default=96)
    p.add_argument("--crop-width", type=int, default=320)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
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

    # Pass 1: index every labeled plate (labels only, cheap). key = (img_name, line_idx).
    plates = {}
    for img_name in images:
        lbl = label_for_image(img_name, name_set)
        if not lbl:
            continue
        try:
            txt = zf.read(lbl).decode("utf-8", "ignore")
        except Exception:
            continue
        for li, line in enumerate(txt.splitlines()):
            parsed = parse_line(line)
            if not parsed:
                continue
            corners, ptype, mode, content = parsed
            if not content or content[0] not in PROVINCES:
                continue
            plates[(img_name, li)] = (img_name, li, corners, ptype, mode, content)

    # Selection: always keep rare TYPES (white, yellow_double); cap each PROVINCE.
    chosen: dict = {}
    for key, rec in plates.items():
        if rec[4] == "double_layer" or rec[3] == "white":
            chosen[key] = rec

    by_prov = collections.defaultdict(list)
    for key, rec in plates.items():
        by_prov[rec[5][0]].append(key)
    for prov, keys in by_prov.items():
        random.shuffle(keys)
        for key in keys[: args.cap_prov]:
            chosen[key] = plates[key]

    # Re-split image-disjoint 70/15/15 within each province stratum.
    img_plates = collections.defaultdict(list)
    for rec in chosen.values():
        img_plates[rec[0]].append(rec)
    img_prov = {img: recs[0][5][0] for img, recs in img_plates.items()}

    prov_imgs = collections.defaultdict(list)
    for img, prov in img_prov.items():
        prov_imgs[prov].append(img)

    img_split = {}
    for prov, imgs in prov_imgs.items():
        imgs = sorted(imgs)
        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        for i, img in enumerate(imgs):
            img_split[img] = "train" if i < n_tr else ("val" if i < n_tr + n_val else "test")

    # Crop each selected plate (read every scene once).
    records = []
    rec_id = 0
    for img_name, recs in img_plates.items():
        data = np.frombuffer(zf.read(img_name), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        sp = img_split[img_name]
        scene_saved = None
        for (_, li, corners, ptype, mode, content) in recs:
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

    def prov_dist(recs):
        return dict(collections.Counter(r["plate_number"][0] for r in recs).most_common())

    summary = {
        "source": "CRPD_all province-aware re-split",
        "cap_prov": args.cap_prov,
        "total": len(records),
        "by_split": {k: len(v) for k, v in splits.items()},
        "plate_type_counts": dict(collections.Counter(r["plate_type"] for r in records)),
        "mode_counts": dict(collections.Counter(r["mode"] for r in records)),
        "province_counts_total": prov_dist(records),
        "province_counts_train": prov_dist(splits["train"]),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
