#!/usr/bin/env python
"""Convert plate manifests to a YOLOv8-POSE dataset (box + 4 corner keypoints).

The 4 keypoints let inference recover the plate quadrilateral and do a perspective
correction before OCR, so the OCR input matches the warped crops used in training.

Keypoint order (after order_points): 0=TL, 1=TR, 2=BR, 3=BL.

Label line format (ultralytics pose):
    cls cx cy w h  px0 py0 v0  px1 py1 v1  px2 py2 v2  px3 py3 v3
all normalized to [0,1] by image width/height; v=2 means visible.

Classes (same as detection):
    0 blue  1 green  2 yellow_single  3 yellow_double  4 white  5 black
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

YOLO_CLASSES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]
CLASS_INDEX = {name: i for i, name in enumerate(YOLO_CLASSES)}


def plate_type_to_class(plate_type: str, mode: str):
    pt = plate_type.lower()
    md = mode.lower()
    if pt == "blue":
        return CLASS_INDEX["blue"]
    if pt == "green":
        return CLASS_INDEX["green"]
    if pt == "yellow":
        return CLASS_INDEX["yellow_double"] if "double" in md else CLASS_INDEX["yellow_single"]
    if pt == "white":
        return CLASS_INDEX["white"]
    if pt == "black":
        return CLASS_INDEX["black"]
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--output-dir", default="datasets/yolo_pose")
    p.add_argument("--symlink", action="store_true")
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
    out = Path(args.output_dir)
    img_dir = out / "images" / args.split
    lbl_dir = out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for manifest in args.manifests:
        for rec in read_jsonl(manifest):
            scene = rec.get("scene_path")
            corners = rec.get("corners")
            box = rec.get("box_xywh")
            if not scene or not Path(scene).is_file() or not corners or len(corners) != 4 or not box:
                skipped += 1
                continue
            cls = plate_type_to_class(rec.get("plate_type", "blue"), rec.get("mode", "single_layer"))
            if cls is None:
                skipped += 1
                continue
            img = cv2.imread(scene)
            if img is None:
                skipped += 1
                continue
            ih, iw = img.shape[:2]

            ordered = order_points(np.asarray(corners, dtype=np.float32))
            x, y, w, h = box
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            cx, cy, nw, nh = [max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh)]
            if nw < 0.001 or nh < 0.001:
                skipped += 1
                continue

            kpt_terms = []
            for (px, py) in ordered:
                npx = max(0.0, min(1.0, float(px) / iw))
                npy = max(0.0, min(1.0, float(py) / ih))
                kpt_terms.append(f"{npx:.6f} {npy:.6f} 2")

            rec_id = rec.get("id", Path(scene).stem)
            dest_img = img_dir / f"{rec_id}.jpg"
            dest_lbl = lbl_dir / f"{rec_id}.txt"
            if args.symlink:
                if not dest_img.exists():
                    dest_img.symlink_to(Path(scene).resolve())
            else:
                shutil.copy2(scene, dest_img)
            line = f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_terms)
            dest_lbl.write_text(line + "\n", encoding="utf-8")
            written += 1

    print(f"Split={args.split}: written={written} skipped={skipped}")

    yaml_path = out / "plate_pose.yaml"
    if not yaml_path.exists():
        content = (
            f"path: {out.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"kpt_shape: [4, 3]\n"
            f"flip_idx: [1, 0, 3, 2]\n\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )
        yaml_path.write_text(content, encoding="utf-8")
        print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
