#!/usr/bin/env python
"""Convert plate OCR manifests to YOLO detection dataset format.

Reads one or more JSONL manifests that contain `scene_path` and `box_xywh` fields,
and produces a YOLO-format dataset (images + labels directories + data.yaml).

Supported YOLO classes:
    0  blue
    1  green
    2  yellow_single
    3  yellow_double
    4  white
    5  black
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


YOLO_CLASSES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]
CLASS_INDEX = {name: i for i, name in enumerate(YOLO_CLASSES)}


def plate_type_to_class(plate_type: str, mode: str) -> int | None:
    pt = plate_type.lower()
    md = mode.lower()
    if pt == "blue":
        return CLASS_INDEX["blue"]
    if pt == "green":
        return CLASS_INDEX["green"]
    if pt == "yellow":
        if "double" in md:
            return CLASS_INDEX["yellow_double"]
        return CLASS_INDEX["yellow_single"]
    if pt == "white":
        return CLASS_INDEX["white"]
    if pt == "black":
        return CLASS_INDEX["black"]
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="JSONL manifests to include (can pass multiple files).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        help="Which split this manifest belongs to.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/yolo",
        help="Root directory for the YOLO dataset.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (faster, less disk).",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_image_dims(path: str | Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img.shape[1], img.shape[0]  # width, height


def box_xywh_to_yolo(box: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    split = args.split
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    for manifest_path in args.manifests:
        records = read_jsonl(manifest_path)
        for rec in records:
            scene_path = rec.get("scene_path")
            if not scene_path or not Path(scene_path).is_file():
                skipped += 1
                continue

            box_xywh = rec.get("box_xywh")
            if not box_xywh or len(box_xywh) < 4:
                skipped += 1
                continue

            plate_type = rec.get("plate_type", "blue")
            mode = rec.get("mode", "single_layer")
            cls_id = plate_type_to_class(plate_type, mode)
            if cls_id is None:
                skipped += 1
                continue

            try:
                img_w, img_h = get_image_dims(scene_path)
            except Exception:
                skipped += 1
                continue

            cx, cy, nw, nh = box_xywh_to_yolo(box_xywh, img_w, img_h)
            if nw < 0.001 or nh < 0.001:
                skipped += 1
                continue

            rec_id = rec.get("id", Path(scene_path).stem)
            dest_img = img_dir / f"{rec_id}.jpg"
            dest_lbl = lbl_dir / f"{rec_id}.txt"

            if args.symlink:
                if not dest_img.exists():
                    dest_img.symlink_to(Path(scene_path).resolve())
            else:
                shutil.copy2(scene_path, dest_img)

            with dest_lbl.open("w", encoding="utf-8") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            written += 1

    print(f"Split={split}: written={written}, skipped={skipped}")

    yaml_path = output_dir / "plate_detect.yaml"
    if not yaml_path.exists():
        yaml_content = (
            f"path: {output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
