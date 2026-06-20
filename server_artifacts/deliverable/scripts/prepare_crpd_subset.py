#!/usr/bin/env python
"""Prepare a CRPD subset from the official archive."""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

from plate_course.visualization import save_overview

TYPE_MAP = {
    "0": ("blue", "single_layer"),
    "1": ("yellow", "single_layer"),
    "2": ("yellow", "double_layer"),
    "3": ("white", "single_layer"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/archives/CRPD_single_google_drive.bin")
    parser.add_argument("--output-dir", default="data/processed/crpd_single_subset")
    parser.add_argument("--max-images", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--crop-width", type=int, default=320)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def find_image_label_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Pair images to labels by structural path (/images/ -> /labels/).

    CRPD_all has subsets (single/double/multi) whose file stems collide across
    subsets (both double and multi use the "37_" prefix), so pairing by bare stem
    would mismatch labels to the wrong image. We instead map each image's full path
    from its images/ directory to the sibling labels/ directory with a .txt suffix.
    """
    name_set = set(names)
    images = [n for n in names if is_image(n)]
    pairs = []
    for image_name in images:
        label_name = None
        for img_seg, lbl_seg in (("/images/", "/labels/"), ("/image/", "/label/")):
            if img_seg in image_name:
                candidate = image_name.replace(img_seg, lbl_seg)
                candidate = candidate.rsplit(".", 1)[0] + ".txt"
                if candidate in name_set:
                    label_name = candidate
                    break
        if label_name is None:
            sibling = image_name.rsplit(".", 1)[0] + ".txt"
            if sibling in name_set:
                label_name = sibling
        if label_name:
            pairs.append((image_name, label_name))
    return pairs


def parse_label_line(line: str):
    raw = line.strip()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in raw.split() if p.strip()]
    # CRPD label: 8 corner coords + type [+ optional plate text].
    # Some plates in multi-plate scenes are box/type-only (no readable text).
    if len(parts) < 9:
        raise ValueError(f"Unexpected CRPD label line: {line!r}")
    nums = [int(float(v)) for v in parts[:8]]
    corners = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
    plate_type, mode = TYPE_MAP.get(parts[8], ("unknown", "unknown"))
    content = "".join(parts[9:]) if len(parts) > 9 else ""
    return corners, plate_type, mode, content


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
        names = zf.namelist()
        pairs = find_image_label_pairs(names)
        random.shuffle(pairs)
        if args.max_images:
            pairs = pairs[: args.max_images]

        rec_id = 0
        for image_member, label_member in pairs:
            data = np.frombuffer(zf.read(image_member), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            text = zf.read(label_member).decode("utf-8", errors="ignore")
            # CRPD-single usually has one plate, but keeping all lines makes the
            # same converter work for multi-plate archives.
            for line_idx, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                try:
                    corners, plate_type, mode, content = parse_label_line(line)
                except Exception:
                    continue
                # Box/type-only plates (no readable text) are unusable for OCR.
                if not content:
                    continue
                crop = crop_plate(img, corners, args.crop_width, args.crop_height)
                image_path = image_dir / f"{rec_id:06d}.jpg"
                crop_path = crop_dir / f"{rec_id:06d}.jpg"
                cv2.imwrite(str(image_path), img)
                cv2.imwrite(str(crop_path), crop)
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                records.append(
                    {
                        "id": f"crpd_{rec_id:06d}",
                        "image_path": str(crop_path.resolve()),
                        "scene_path": str(image_path.resolve()),
                        "plate_number": content,
                        "plate_type": plate_type,
                        "mode": mode,
                        "source": "CRPD",
                        "corners": corners,
                        "box_xywh": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                        "archive_member": image_member,
                        "label_member": label_member,
                        "line_index": line_idx,
                    }
                )
                rec_id += 1

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

    save_overview(records, figure_dir / "crpd_subset_overview.jpg", max_items=24)
    type_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for record in records:
        type_counts[record["plate_type"]] = type_counts.get(record["plate_type"], 0) + 1
        mode_counts[record["mode"]] = mode_counts.get(record["mode"], 0) + 1
    summary = {
        "source": "CRPD",
        "archive": str(Path(args.archive).resolve()),
        "count": len(records),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "plate_type_counts": type_counts,
        "mode_counts": mode_counts,
        "note": "GT plate crops are generated by perspective transform from CRPD quadrilateral annotations.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
