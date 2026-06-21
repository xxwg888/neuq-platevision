#!/usr/bin/env python
"""Single-image traditional-baseline inference API: template-matching (NCC) + HOG+KNN.

A real per-image backend interface (not a manifest evaluator). Reuses the EXACT
segmentation / HOG / NCC code as scripts/knn_baseline.py + scripts/template_baseline.py
so results reproduce the reported baselines. CPU-only; torch not required.

Two inputs (auto-detected, or forced via --input-type):
  * 整车图  -> YOLO-pose ONNX(CPU)检测角点 -> 透视校正 -> 传统 OCR
  * 已校正车牌图 / crop -> 跳过检测,直接传统 OCR

CLI
---
python scripts/traditional_infer_demo.py \
    --image xxx.jpg --mode both \
    --yolo-onnx models/yolo_pose/best.onnx \
    --templates models/template/templates.npz \
    --knn-model models/knn/knn_baseline.xml \
    --vis-dir outputs/demo_vis

JSON (stdout) fields: input_path, is_car_image, used_detector, plate_type, bbox,
corners, ncc_text, knn_text, per_char[{index,bbox,ncc,ncc_similarity,knn,
knn_confidence,knn_neighbor_dist}], vis_paths{detected,plate_crop,binary,
segmented,classification_vis}.

Windows 中文路径:图像读写走 cv2.imdecode(np.fromfile) / cv2.imencode(...).tofile()。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

from knn_baseline import (  # noqa: E402
    IDX_TO_CHAR, binarize, hog_features, segment_runs,
)
from template_baseline import ncc_vector, resized_glyph  # noqa: E402

TRIM_X, TRIM_Y = 0.02, 0.10  # must match knn_baseline.binarize border trim


# --------------------------------------------------------- unicode-safe IO ---
def imread_u(path):
    """cv2.imread that tolerates non-ASCII (e.g. Chinese) paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    path = str(path)
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# --------------------------------------------------------------- CJK font ----
def _cjk_font_path():
    here = Path(__file__).resolve().parent
    cands = [here.parent / "fonts" / "NotoSansCJK-Regular.ttc",
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
             Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
    return next((str(p) for p in cands if p.is_file()), None)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp = _cjk_font_path()
    if fp:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def put_cjk(img_bgr, text, xy, color=(0, 255, 0), size=22):
    from PIL import Image, ImageDraw, ImageFont
    fp = _cjk_font_path()
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, fill=color[::-1], font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------- classifiers ---
def ncc_classify(patch, classes, templates):
    v = ncc_vector(resized_glyph(patch)).astype(np.float32)
    scores = templates @ v
    j = int(np.argmax(scores))
    return str(classes[j]), float(scores[j])  # char, cosine similarity in [-1,1]


def knn_classify(patch, knn, k):
    feat = np.array([hog_features(patch)], dtype=np.float32)
    _, results, neighbours, dists = knn.findNearest(feat, k=k)
    win = int(results[0][0])
    votes = int(np.sum(neighbours[0].astype(int) == win))
    return IDX_TO_CHAR.get(win, "?"), round(votes / k, 4), round(float(np.mean(dists[0])), 2)


# --------------------------------------------------------------- detection ---
def detect_and_warp(img_bgr, yolo_onnx, conf, plate_w, plate_h):
    from onnx_infer_demo import detect_pose_onnx, warp_plate
    import onnxruntime as ort
    sess = ort.InferenceSession(yolo_onnx, providers=["CPUExecutionProvider"])
    dets = detect_pose_onnx(sess, img_bgr, conf_thr=conf)
    if not dets:
        return None
    best = max(dets, key=lambda d: d["confidence"])
    crop = warp_plate(img_bgr, best["corners"], plate_w, plate_h) if best.get("corners") else None
    best["crop"] = crop
    return best


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["ncc", "knn", "both"], default="both")
    ap.add_argument("--input-type", choices=["auto", "car", "plate"], default="auto")
    ap.add_argument("--yolo-onnx", default="models/yolo_pose/best.onnx")
    ap.add_argument("--templates", default="models/template/templates.npz")
    ap.add_argument("--knn-model", default="models/knn/knn_baseline.xml")
    ap.add_argument("--vis-dir", default="outputs/demo_vis")
    ap.add_argument("--n-chars", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--plate-width", type=int, default=160)
    ap.add_argument("--plate-height", type=int, default=48)
    args = ap.parse_args()

    pw, ph = args.plate_width, args.plate_height
    img = imread_u(args.image)
    if img is None:
        print(json.dumps({"input_path": args.image, "error": "cannot_read_image"}, ensure_ascii=False))
        return
    ih, iw = img.shape[:2]

    # ---- decide car vs plate, get rectified plate + (optional) detection ----
    used_detector = False
    is_car_image = False
    bbox = None
    corners = None
    plate_type = None
    det = None
    want_detect = args.input_type in ("auto", "car") and Path(args.yolo_onnx).is_file()
    if want_detect:
        det = detect_and_warp(img, args.yolo_onnx, args.conf, pw, ph)
        if det is not None and det.get("crop") is not None:
            used_detector = True
            area_ratio = (det["box"][2] * det["box"][3]) / float(iw * ih)
            # a full-frame "detection" means the input was already a plate crop
            if args.input_type == "car" or area_ratio < 0.85:
                is_car_image = True
                plate = det["crop"]
                bbox = [round(float(v), 1) for v in det["box"]]
                corners = [[round(float(c[0]), 1), round(float(c[1]), 1)] for c in det["corners"]]
                plate_type = det.get("plate_type")
            else:
                plate = cv2.resize(img, (pw, ph))
        else:
            plate = cv2.resize(img, (pw, ph))
    else:
        plate = cv2.resize(img, (pw, ph))

    # ---- n_chars heuristic ----
    n_chars = args.n_chars
    if n_chars is None:
        n_chars = 8 if plate_type == "green" else (7 if plate_type else None)

    # ---- load classifiers ----
    td = np.load(args.templates, allow_pickle=True)
    classes, templates = td["classes"], td["templates"].astype(np.float32)
    knn = None
    if args.mode in ("knn", "both"):
        knn = cv2.ml.KNearest_load(args.knn_model)
        knn.setDefaultK(args.k)

    # ---- segment (with coords) on the rectified plate ----
    bw = binarize(plate)
    runs = segment_runs(bw, n_chars)
    offx = int(round(pw * TRIM_X))
    offy = int(round(ph * TRIM_Y))

    per_char = []
    ncc_text, knn_text = [], []
    for i, (s, e) in enumerate(runs):
        patch = bw[:, s:e]
        rec = {"index": i, "bbox": [offx + int(s), offy, int(e - s), bw.shape[0]]}
        if args.mode in ("ncc", "both"):
            c, sim = ncc_classify(patch, classes, templates)
            rec["ncc"] = c; rec["ncc_similarity"] = round(sim, 4); ncc_text.append(c)
        if args.mode in ("knn", "both"):
            c, conf, dist = knn_classify(patch, knn, args.k)
            rec["knn"] = c; rec["knn_confidence"] = conf; rec["knn_neighbor_dist"] = dist; knn_text.append(c)
        per_char.append(rec)

    # ---- visualizations ----
    vis_dir = Path(args.vis_dir); vis_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    vis = {"detected": None, "plate_crop": None, "binary": None, "segmented": None, "classification_vis": None}

    if used_detector and is_car_image and det is not None:
        canvas = img.copy()
        x, y, w, h = [int(v) for v in det["box"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 210, 255), 3)
        for cx, cy in det["corners"]:
            cv2.circle(canvas, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        canvas = put_cjk(canvas, f'{plate_type}', (x, max(0, y - 28)), (0, 255, 0), 26)
        p = vis_dir / f"{stem}_detected.png"; imwrite_u(p, canvas); vis["detected"] = str(p)

    p = vis_dir / f"{stem}_plate_crop.png"; imwrite_u(p, plate); vis["plate_crop"] = str(p)
    p = vis_dir / f"{stem}_binary.png"; imwrite_u(p, bw); vis["binary"] = str(p)

    seg_canvas = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for (s, e) in runs:
        cv2.rectangle(seg_canvas, (int(s), 0), (int(e), bw.shape[0] - 1), (0, 0, 255), 1)
    seg_canvas = cv2.resize(seg_canvas, (pw * 3, ph * 3), interpolation=cv2.INTER_NEAREST)
    p = vis_dir / f"{stem}_segmented.png"; imwrite_u(p, seg_canvas); vis["segmented"] = str(p)

    # classification figure (matplotlib, CJK)
    plt = _mpl()
    n = max(len(runs), 1)
    fig, axs = plt.subplots(1, n, figsize=(1.5 * n, 2.6))
    if n == 1:
        axs = [axs]
    for i, (s, e) in enumerate(runs):
        axs[i].imshow(bw[:, s:e], cmap="gray"); axs[i].axis("off")
        lines = []
        if args.mode in ("ncc", "both"):
            lines.append(f'NCC:{per_char[i]["ncc"]}({per_char[i]["ncc_similarity"]:.2f})')
        if args.mode in ("knn", "both"):
            lines.append(f'KNN:{per_char[i]["knn"]}({per_char[i]["knn_confidence"]:.2f})')
        agree = (args.mode != "both") or (per_char[i].get("ncc") == per_char[i].get("knn"))
        axs[i].set_title("\n".join(lines), fontsize=10, color=("green" if agree else "red"))
    fig.suptitle("逐字符分类(模板NCC vs HOG+KNN)", fontsize=12, y=1.06)
    fig.tight_layout()
    p = vis_dir / f"{stem}_classification.png"; fig.savefig(str(p), dpi=150, bbox_inches="tight"); plt.close(fig)
    vis["classification_vis"] = str(p)

    out = {
        "input_path": args.image,
        "is_car_image": is_car_image,
        "used_detector": used_detector,
        "plate_type": plate_type,
        "bbox": bbox,
        "corners": corners,
        "n_chars_used": n_chars,
        "ncc_text": "".join(ncc_text) if ncc_text else None,
        "knn_text": "".join(knn_text) if knn_text else None,
        "per_char": per_char,
        "vis_paths": vis,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
