#!/usr/bin/env python
"""Prepare a CCPD/CCPD2019 subset from an archive or extracted folder."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from plate_course.chars import decode_ccpd_plate
from plate_course.visualization import save_overview


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CCPD archive or extracted directory")
    parser.add_argument("--output-dir", default="data/processed/ccpd_blue_subset")
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


def is_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


class ImageSource:
    def __init__(self, input_path: str | Path):
        self.input_path = Path(input_path)
        self.kind = "dir"
        self.zip_file = None
        self.tar_file = None
        if self.input_path.is_file() and zipfile.is_zipfile(self.input_path):
            self.kind = "zip"
            self.zip_file = zipfile.ZipFile(self.input_path)
        elif self.input_path.is_file() and tarfile.is_tarfile(self.input_path):
            self.kind = "tar"
            self.tar_file = tarfile.open(self.input_path)

    def names(self) -> list[str]:
        if self.kind == "zip":
            return [n for n in self.zip_file.namelist() if is_image(n)]
        if self.kind == "tar":
            return [m.name for m in self.tar_file.getmembers() if m.isfile() and is_image(m.name)]
        return [str(p) for p in self.input_path.rglob("*") if p.is_file() and is_image(str(p))]

    def read_image(self, name: str) -> np.ndarray | None:
        if self.kind == "zip":
            data = np.frombuffer(self.zip_file.read(name), dtype=np.uint8)
        elif self.kind == "tar":
            member = self.tar_file.getmember(name)
            f = self.tar_file.extractfile(member)
            if f is None:
                return None
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            data = np.fromfile(name, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def close(self):
        if self.zip_file:
            self.zip_file.close()
        if self.tar_file:
            self.tar_file.close()


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def infer_subset(name: str) -> str:
    parts = Path(name).parts
    for part in parts:
        if part.lower().startswith("ccpd_") or part.lower() in {"train", "val", "test"}:
            return part
    return "unknown"


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

    source = ImageSource(args.input)
    names = sorted(source.names())
    random.shuffle(names)
    if args.max_images:
        names = names[: args.max_images]

    records = []
    for idx, name in enumerate(names):
        try:
            meta = parse_ccpd_name(name)
        except Exception:
            continue
        img = source.read_image(name)
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
                "id": f"ccpd_{idx:06d}",
                "image_path": str(crop_path.resolve()),
                "scene_path": str(image_path.resolve()),
                "plate_number": meta["plate_number"],
                "plate_type": "blue",
                "mode": "single_layer",
                "source": "CCPD",
                "subset": infer_subset(name),
                "corners": meta["corners"],
                "box_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "archive_member": name,
            }
        )
    source.close()

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

    save_overview(records, figure_dir / "ccpd_subset_overview.jpg", max_items=24)
    summary = {
        "source": "CCPD",
        "input": str(Path(args.input).resolve()),
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
