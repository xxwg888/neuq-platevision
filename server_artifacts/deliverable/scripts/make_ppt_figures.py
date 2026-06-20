#!/usr/bin/env python
"""Generate PPT-ready figures: pipeline diagram, perspective-correction demo,
character-segmentation demo, and the CRNN training curves (from train_log.csv).

Outputs into outputs/figures/ppt/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

OUT = PROJ / "outputs" / "figures" / "ppt"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(1); d = np.diff(pts, 1).reshape(-1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 3)
    boxes = [("整车图\ninput", 0.3), ("YOLOv8n-pose\n检测+颜色+4角点", 2.7),
             ("透视校正\nwarp", 5.7), ("CRNN-CTC\n识别", 8.0), ("车牌号\n皖A990H5", 10.4)]
    w = 2.1
    for i, (t, x) in enumerate(boxes):
        c = ["#e8f0fe", "#d2e3fc", "#fce8e6", "#e6f4ea", "#fef7e0"][i]
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.1, boxstyle="round,pad=0.08",
                                    fc=c, ec="#5f6368", lw=1.5))
        ax.text(x + w / 2, 1.55, t, ha="center", va="center", fontsize=12)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.55), (boxes[i + 1][1] - 0.02, 1.55),
                                         arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#5f6368"))
    ax.text(6.5, 2.7, "车牌定位 → 透视校正 → 字符识别(端到端整牌 0.9206 / 字符 0.9852)",
            ha="center", fontsize=12, color="#1a73e8")
    fig.tight_layout(); fig.savefig(OUT / "pipeline_diagram.png", dpi=160); plt.close(fig)


def perspective_demo():
    recs = [json.loads(l) for l in (PROJ / "data/processed/ocr_5color/manifests/test.jsonl").open()]
    # pick a clearly skewed plate: corners deviate most from an axis-aligned rect
    best = None; best_skew = -1
    for r in recs[:1500]:
        cn = r.get("corners")
        if not cn or len(cn) != 4:
            continue
        p = order_points(np.asarray(cn, np.float32))
        skew = abs(p[0][1] - p[1][1]) + abs(p[2][1] - p[3][1])  # top/bottom edge tilt
        if skew > best_skew and Path(r["scene_path"]).exists():
            best_skew = skew; best = r
    if best is None:
        return
    img = cv2.imread(best["scene_path"]); p = order_points(np.asarray(best["corners"], np.float32))
    x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
    pad = 12
    region = img[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)]
    dst = np.float32([[0, 0], [319, 0], [319, 95], [0, 95]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(p, dst), (320, 96))
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
    axs[0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)); axs[0].set_title("原始倾斜车牌区域", fontsize=12)
    axs[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)); axs[1].set_title("4角点透视校正后", fontsize=12)
    for a in axs: a.axis("off")
    fig.suptitle(f'透视校正示例(GT: {best["plate_number"]})', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "perspective_demo.png", dpi=160); plt.close(fig)


def segmentation_demo():
    try:
        from knn_baseline import binarize, segment_characters_projection, load_plate, read_jsonl
    except Exception as e:
        print("seg demo skipped:", e); return
    recs = list(read_jsonl(PROJ / "data/processed/ocr_5color/manifests/test.jsonl"))
    rec = next((r for r in recs if len(r["plate_number"]) == 7 and Path(r["image_path"]).exists()), None)
    if rec is None:
        return
    img = load_plate(rec["image_path"], 160, 48); bw = binarize(img)
    segs = segment_characters_projection(bw, n_chars=len(rec["plate_number"]))  # list of char images
    n = len(segs)
    cols = max(n, 2)
    fig = plt.figure(figsize=(1.5 * cols, 3.4))
    ax0 = fig.add_subplot(3, 1, 1); ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax0.set_title(f'① 校正车牌  (GT: {rec["plate_number"]})', fontsize=11); ax0.axis("off")
    ax1 = fig.add_subplot(3, 1, 2); ax1.imshow(bw, cmap="gray")
    ax1.set_title("② 二值化(边框裁剪+形态学去噪)", fontsize=11); ax1.axis("off")
    for i, seg in enumerate(segs):
        a = fig.add_subplot(3, cols, 2 * cols + i + 1)
        a.imshow(seg, cmap="gray")
        a.set_title(rec["plate_number"][i] if i < len(rec["plate_number"]) else "", fontsize=12)
        a.axis("off")
    fig.suptitle("③ 投影法强制切分到 N 字", fontsize=11, y=0.34)
    fig.tight_layout(); fig.savefig(OUT / "segmentation_demo.png", dpi=160); plt.close(fig)


def crnn_curves():
    f = PROJ / "outputs/models/ocr_5color_prov/train_log.csv"
    if not f.exists():
        return
    rows = list(csv.DictReader(f.open()))
    ep = [int(r["epoch"]) for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train_loss")
    axs[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val_loss")
    axs[0].set_title("CRNN 训练/验证 Loss"); axs[0].set_xlabel("epoch"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(ep, [float(r["val_plate_accuracy"]) for r in rows], label="val 整牌准确率")
    axs[1].plot(ep, [float(r["val_character_accuracy"]) for r in rows], label="val 字符准确率")
    axs[1].set_title("CRNN 验证集准确率"); axs[1].set_xlabel("epoch"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "crnn_training_curves.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    pipeline_diagram(); print("pipeline_diagram.png")
    perspective_demo(); print("perspective_demo.png")
    segmentation_demo(); print("segmentation_demo.png")
    crnn_curves(); print("crnn_training_curves.png")
    print("PPT figures ->", OUT)
