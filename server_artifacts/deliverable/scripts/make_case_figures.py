#!/usr/bin/env python
"""Render success / failure case figures for the detection+recognition pipeline.

For each sampled test scene it runs the real pipeline (YOLO-pose detect + colour +
4 corners -> perspective warp -> CRNN OCR), then draws a panel showing:
  * the scene with the ground-truth box (green), detected box (yellow) and the
    4 predicted corners (red), so the localisation + corner quality is visible;
  * the perspective-corrected crop that is actually fed to the OCR;
  * GT vs predicted plate string + colour, with a ✓ / ✗ verdict.

Panels are stacked into a success montage and a failure montage, balanced across
plate colours so the report shows every supported type.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from infer import OCRRecognizer, YOLODetector, crop_from_box, warp_plate  # noqa: E402

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"


def read_jsonl(path):
    import json
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def compute_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def put_cjk(img_bgr, lines, org, font, line_h, colors):
    """Draw multiple CJK text lines onto a BGR image via PIL; returns BGR."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    x, y = org
    for text, color in zip(lines, colors):
        draw.text((x, y), text, font=font, fill=color)  # color is RGB
        y += line_h
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def make_panel(scene, gt_box, det, gt_str, pred_str, gt_type, font, panel_w=900):
    """One case panel: annotated scene (left) + warped crop & text (right)."""
    img = scene.copy()
    # GT box (green)
    gx, gy, gw, gh = [int(v) for v in gt_box]
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 200, 0), 3)
    crop = None
    if det is not None:
        bx, by, bw, bh = [int(v) for v in det["box"]]
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
        corners = det.get("corners")
        if corners and len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            for p in pts:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            crop = warp_plate(scene, corners, 320, 96)
        else:
            crop = crop_from_box(scene, det["box"], 320, 96)

    # Left: scene scaled to fixed height
    H = 300
    sh, sw = img.shape[:2]
    scale = H / sh
    left = cv2.resize(img, (int(sw * scale), H))
    left = left[:, : min(left.shape[1], panel_w // 2)]

    # Right: white canvas with crop + text
    right = np.full((H, panel_w - left.shape[1], 3), 255, np.uint8)
    if crop is not None:
        right[15:15 + 96, 20:20 + 320] = crop
    ok = pred_str == gt_str
    lines = [
        f"GT  : {gt_str}  [{gt_type}]",
        f"Pred: {pred_str}",
        "正确 ✓" if ok else "错误 ✗",
    ]
    colors = [(0, 0, 0), (0, 0, 0), (0, 160, 0) if ok else (220, 0, 0)]
    right = put_cjk(right, lines, (20, 130), font, 40, colors)

    panel = np.hstack([left, right])
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-model", default="outputs/yolo_pose/best.pt")
    ap.add_argument("--ocr-checkpoint", default="outputs/models/ocr_best.pt")
    ap.add_argument("--test-manifest", default="data/processed/ocr_5color/manifests/test.jsonl")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-type", type=int, default=2, help="success/failure cases per colour")
    ap.add_argument("--scan-limit", type=int, default=1200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(CJK_FONT, 30)

    recog = OCRRecognizer(args.ocr_checkpoint, args.device, (48, 160))
    det = YOLODetector(args.yolo_model, conf=0.25, iou=0.45, imgsz=640, device=args.device)

    records = read_jsonl(args.test_manifest)
    # spread across colours
    import random
    random.seed(7)
    random.shuffle(records)

    succ = defaultdict(list)
    fail = defaultdict(list)
    need = args.per_type
    scanned = 0
    for rec in records:
        if scanned >= args.scan_limit:
            break
        ptype = rec.get("plate_type", "unknown")
        if len(succ[ptype]) >= need and len(fail[ptype]) >= need:
            continue
        scene = cv2.imread(rec["scene_path"])
        if scene is None:
            continue
        scanned += 1
        gt_box = rec["box_xywh"]
        dets = det.detect(scene)
        best, best_iou = None, 0.0
        for d in dets:
            iou = compute_iou(d["box"], gt_box)
            if iou > best_iou:
                best, best_iou = d, iou
        if best is None or best_iou < 0.4:
            continue
        if best.get("corners") and len(best["corners"]) == 4:
            crop = warp_plate(scene, best["corners"], 160, 48)
        else:
            crop = crop_from_box(scene, best["box"], 160, 48)
        pred, _ = recog.recognize(crop)
        gt = rec["plate_number"]
        bucket = succ if pred == gt else fail
        if len(bucket[ptype]) < need:
            bucket[ptype].append(make_panel(scene, gt_box, best, gt, pred, ptype, font))

    def montage(buckets, title, path):
        panels = [p for t in sorted(buckets) for p in buckets[t]]
        if not panels:
            print(f"no panels for {title}")
            return
        w = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 6, 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=(230, 230, 230)) for p in panels]
        grid = np.vstack(panels)
        header = np.full((46, grid.shape[1], 3), 255, np.uint8)
        header = put_cjk(header, [title], (16, 6), font, 36, [(0, 0, 0)])
        out = np.vstack([header, grid])
        cv2.imwrite(str(path), out)
        print(f"saved {path}  ({len(panels)} cases)")

    montage(succ, "识别成功案例 (绿框=真值, 黄框=检测, 红=4角点)", out_dir / "success_cases.png")
    montage(fail, "识别失败案例 (绿框=真值, 黄框=检测, 红=4角点)", out_dir / "failure_cases.png")


if __name__ == "__main__":
    main()
