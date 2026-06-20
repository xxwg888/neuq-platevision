#!/usr/bin/env python
"""Prepare a small CCPD-Green subset directly from the downloaded zip."""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

from plate_course.chars import decode_ccpd_plate
from plate_course.visualization import save_overview


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/archives/CCPD_Green_google_drive.bin")
    parser.add_argument("--output-dir", default="data/processed/ccpd_green_subset")
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--crop-width", type=int, default=320)
    return parser.parse_args()


def parse_point(text: str) -> list[int]:
    x, y = text.split("&")
    return [int(x), int(y)]


def parse_ccpd_name(name: str) -> dict:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected CCPD filename: {name}")
    bbox = [parse_point(p) for p in parts[2].split("_")]
    corners = [parse_point(p) for p in parts[3].split("_")]
    label_indices = [int(v) for v in parts[4].split("_")]
    return {
        "bbox": bbox,
        "corners": corners,
        "plate_number": decode_ccpd_plate(label_indices),
        "label_indices": label_indices,
    }


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def crop_plate(img: np.ndarray, corners: list[list[int]], width: int, height: int) -> np.ndarray:
    src = order_points(np.asarray(corners, dtype=np.float32))
    dst = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (width, height))


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    crop_dir = output_dir / "crops"
    manifest_dir = output_dir / "manifests"
    figure_dir = output_dir / "figures"
    for path in [image_dir, crop_dir, manifest_dir, figure_dir]:
        path.mkdir(parents=True, exist_ok=True)

    records = []
    with zipfile.ZipFile(args.archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        names = sorted(names)
        if args.max_images:
            train_names = [n for n in names if "/train/" in n]
            val_names = [n for n in names if "/val/" in n]
            test_names = [n for n in names if "/test/" in n]
            other_names = [n for n in names if n not in set(train_names + val_names + test_names)]
            sampled: list[str] = []
            groups = [train_names, val_names, test_names, other_names]
            weights = [0.7, 0.15, 0.15, 0.0]
            for group, weight in zip(groups, weights):
                if not group or weight <= 0:
                    continue
                k = min(len(group), max(1, int(args.max_images * weight)))
                sampled.extend(random.sample(group, k))
            if len(sampled) < args.max_images:
                remaining = [n for n in names if n not in set(sampled)]
                sampled.extend(random.sample(remaining, min(args.max_images - len(sampled), len(remaining))))
            names = sampled[: args.max_images]
            random.shuffle(names)
        for idx, name in enumerate(names):
            meta = parse_ccpd_name(name)
            data = np.frombuffer(zf.read(name), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            crop = crop_plate(img, meta["corners"], args.crop_width, args.crop_height)
            image_path = image_dir / f"{idx:06d}.jpg"
            crop_path = crop_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(image_path), img)
            cv2.imwrite(str(crop_path), crop)
            xs = [p[0] for p in meta["corners"]]
            ys = [p[1] for p in meta["corners"]]
            records.append(
                {
                    "id": f"ccpd_green_{idx:06d}",
                    "image_path": str(crop_path.resolve()),
                    "scene_path": str(image_path.resolve()),
                    "plate_number": meta["plate_number"],
                    "plate_type": "green",
                    "mode": "new_energy",
                    "source": "CCPD-Green",
                    "corners": meta["corners"],
                    "box_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "archive_member": name,
                }
            )

    random.shuffle(records)
    n_train = int(len(records) * args.train_ratio)
    n_val = int(len(records) * args.val_ratio)
    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
        "all": records,
    }
    for split, split_records in splits.items():
        write_jsonl(manifest_dir / f"{split}.jsonl", split_records)

    save_overview(records, figure_dir / "ccpd_green_subset_overview.jpg", max_items=24)
    summary = {
        "source": "CCPD-Green",
        "archive": str(Path(args.archive).resolve()),
        "count": len(records),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "note": "GT plate crops are generated by perspective transform from CCPD filename corner annotations.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
