#!/usr/bin/env python
"""Self-contained CPU-only ONNX inference demo: YOLOv8n-pose ONNX + CRNN-CTC ONNX.

Proves both exported ONNX models run end-to-end on plain `onnxruntime` CPU, with
full keypoint decoding (for perspective warp) and CTC greedy decoding — no torch,
no ultralytics required at inference time.

Usage:
    python scripts/onnx_infer_demo.py car.jpg \
        --pose-onnx outputs/yolo_pose/best.onnx \
        --ocr-onnx  outputs/models/ocr_best.onnx \
        --charset   deliverable/charset.txt \
        --vis-out   out_vis.png

ONNX tensor contracts
---------------------
YOLOv8n-pose : input  'images' float32 [1, 3, 640, 640], RGB, value = pixel/255
               output 'output0' float32 [1, 22, 8400]
                 channel layout (22) = 4 box(cx,cy,w,h, 640-space)
                                     + 6 class scores (blue/green/yellow_single/
                                       yellow_double/white/black)
                                     + 4 keypoints x (x, y, conf)  -> 12
                 8400 = anchor predictions; needs transpose + conf-threshold + NMS,
                 then scale boxes & keypoints back to original image size.
CRNN-CTC     : input  'input'  float32 [batch, 3, 48, 160], RGB,
                       value = (pixel/255 - 0.5) / 0.5
               output 'logits' float32 [batch, 40, 76]  (T=40 timesteps, 76 classes)
                 CTC greedy decode: argmax over 76 per timestep -> collapse
                 consecutive repeats -> drop blank (index 0) -> map to charset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]


def load_charset(path: str) -> list[str]:
    # one token per line; line 0 is the blank token. Keep order = class index.
    return [ln.rstrip("\n") for ln in Path(path).read_text(encoding="utf-8").splitlines()]


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def warp_plate(img, corners, w=160, h=48):
    src = order_points(np.asarray(corners, dtype=np.float32))
    dst = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def detect_pose_onnx(sess, img_bgr, imgsz=640, conf_thr=0.25, iou_thr=0.45):
    ih, iw = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (imgsz, imgsz)).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]  # [1,3,640,640]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # [1,22,8400]
    pred = np.squeeze(out, 0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)  # -> [8400, 22]

    box = pred[:, :4]
    cls = pred[:, 4:4 + len(CLASS_NAMES)]
    kpt = pred[:, 4 + len(CLASS_NAMES):]  # [N, 12]
    cls_id = cls.argmax(1)
    score = cls.max(1)
    keep = score >= conf_thr
    box, kpt, cls_id, score = box[keep], kpt[keep], cls_id[keep], score[keep]
    if len(box) == 0:
        return []

    sx, sy = iw / imgsz, ih / imgsz
    nms_boxes, meta = [], []
    for (cx, cy, w, h), k, cid, sc in zip(box, kpt, cls_id, score):
        x, y = (cx - w / 2) * sx, (cy - h / 2) * sy
        nms_boxes.append([float(x), float(y), float(w * sx), float(h * sy)])
        corners = [[float(k[3 * j] * sx), float(k[3 * j + 1] * sy)] for j in range(4)]
        meta.append((float(sc), int(cid), corners))

    idxs = cv2.dnn.NMSBoxes(nms_boxes, [m[0] for m in meta], conf_thr, iou_thr)
    dets = []
    for i in np.array(idxs).flatten() if len(idxs) else []:
        sc, cid, corners = meta[i]
        dets.append({"box": nms_boxes[i], "confidence": round(sc, 4),
                     "plate_type": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "unknown",
                     "corners": corners})
    return dets


def ctc_greedy_decode(logits, charset):
    # logits: [1, T, C] -> string
    seq = logits[0].argmax(-1).tolist()
    out, prev = [], -1
    for idx in seq:
        if idx != prev and idx != 0:   # collapse repeats, drop blank(0)
            out.append(charset[idx])
        prev = idx
    return "".join(out)


def recognize_ocr_onnx(sess, crop_bgr, charset):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (160, 48)).astype(np.float32) / 255.0
    r = (r - 0.5) / 0.5
    inp = np.transpose(r, (2, 0, 1))[None].astype(np.float32)  # [1,3,48,160]
    logits = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # [1,40,76]
    return ctc_greedy_decode(logits, charset)


def draw_vis(img_bgr, plates, out_path):
    from PIL import Image, ImageDraw, ImageFont
    _here = Path(__file__).resolve().parent
    fonts = [str(_here.parent / "fonts" / "NotoSansCJK-Regular.ttc"),
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
             "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
    font = next((ImageFont.truetype(f, max(18, img_bgr.shape[0] // 30))
                 for f in fonts if Path(f).is_file()), ImageFont.load_default())
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for p in plates:
        x, y, w, h = p["box"]
        d.rectangle([x, y, x + w, y + h], outline=(255, 210, 0), width=3)
        for cx, cy in p["corners"]:
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
        d.text((x, max(0, y - font.size - 6)),
               f'{p["plate_number"]} ({p["plate_type"]})', fill=(0, 255, 0), font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--pose-onnx", default="outputs/yolo_pose/best.onnx")
    ap.add_argument("--ocr-onnx", default="outputs/models/ocr_best.onnx")
    ap.add_argument("--charset", default="deliverable/charset.txt")
    ap.add_argument("--vis-out", default=None)
    args = ap.parse_args()

    prov = ["CPUExecutionProvider"]  # force CPU to demonstrate CPU capability
    pose = ort.InferenceSession(args.pose_onnx, providers=prov)
    ocr = ort.InferenceSession(args.ocr_onnx, providers=prov)
    charset = load_charset(args.charset)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")

    plates = []
    for det in detect_pose_onnx(pose, img):
        crop = warp_plate(img, det["corners"]) if det["corners"] else img
        det["plate_number"] = recognize_ocr_onnx(ocr, crop, charset)
        det["box"] = [round(v, 1) for v in det["box"]]
        det["corners"] = [[round(c[0], 1), round(c[1], 1)] for c in det["corners"]]
        plates.append(det)

    result = {"image": args.image, "backend": "onnxruntime-CPU", "plates": plates}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.vis_out:
        draw_vis(img, plates, args.vis_out)
        print(f"visualization saved to {args.vis_out}")


if __name__ == "__main__":
    main()
