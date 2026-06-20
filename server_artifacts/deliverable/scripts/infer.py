#!/usr/bin/env python
"""Full license plate inference pipeline: YOLO detection + CRNN/LPRNet OCR.

Usage (single image):
    python scripts/infer.py image.jpg \
        --yolo-model outputs/yolo/best.pt \
        --ocr-checkpoint outputs/models/crnn_lite_mixed_current_e40/best.pt

Usage (batch, JSON-Lines output):
    python scripts/infer.py img1.jpg img2.jpg --output results.jsonl

Output per image (JSON):
{
  "image_path": "...",
  "plates": [
    {
      "plate_number": "浙A809JC",
      "plate_type": "blue",
      "confidence": 0.96,
      "box": [x, y, w, h],
      "corners": null,
      "characters": [{"text": "浙", "confidence": 0.94}, ...],
      "valid": true,
      "rule_reason": "ok",
      "elapsed_ms": 23.4
    }
  ],
  "elapsed_ms": 25.1
}

If YOLO model is not provided, the script falls back to OpenCV HSV-based plate localization
and runs OCR on the detected region.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from plate_course.chars import decode_indices, greedy_decode
from plate_course.dataset import preprocess_plate_image
from plate_course.postprocess import infer_plate_type_from_length, validate_china_plate
from plate_course.runtime import build_model_from_checkpoint, get_checkpoint_image_size


YOLO_CLASS_NAMES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]


def parse_args():
    parser = argparse.ArgumentParser(description="License plate detection and recognition pipeline.")
    parser.add_argument("images", nargs="+", help="Input image paths.")
    parser.add_argument("--yolo-model", default=None, help="YOLO .pt or .onnx checkpoint path.")
    parser.add_argument("--ocr-checkpoint", default="outputs/models/crnn_lite_mixed_current_e40/best.pt")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--crop-height", type=int, default=48)
    parser.add_argument("--crop-width", type=int, default=160)
    parser.add_argument("--output", default=None, help="Optional output JSONL path.")
    parser.add_argument("--vis-dir", default=None,
                        help="If set, save an annotated visualization PNG per image into this directory.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


# ---- visualization (box + corners + recognized text, CJK-safe) -------------
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _load_cjk_font(size: int):
    from PIL import ImageFont
    for fp in _CJK_FONT_CANDIDATES:
        if Path(fp).is_file():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def render_visualization(img_path: str, result: dict, out_path: Path) -> None:
    """Draw detection box (yellow), 4 corners (red), and recognized plate text
    (CJK via PIL) onto the image, and save to out_path."""
    from PIL import Image, ImageDraw
    img = cv2.imread(img_path)
    if img is None:
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_cjk_font(max(18, img.shape[0] // 30))
    for p in result.get("plates", []):
        x, y, w, h = p["box"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 210, 0), width=3)
        if p.get("corners"):
            for cx, cy in p["corners"]:
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
        label = f'{p["plate_number"]} ({p["plate_type"]} {p["confidence"]:.2f})'
        ty = max(0, y - (font.size + 6))
        try:
            tb = draw.textbbox((x, ty), label, font=font)
            draw.rectangle(tb, fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((x, ty), label, fill=(0, 255, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(out_path))


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_plate(img: np.ndarray, corners: list, width: int, height: int) -> np.ndarray:
    src = order_points(np.asarray(corners, dtype=np.float32))
    dst = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (width, height))


def crop_from_box(img: np.ndarray, box: list[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    ih, iw = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, iw - x)
    h = min(h, ih - y)
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


def opencv_detect_plates(img: np.ndarray) -> list[dict]:
    """Fallback: OpenCV HSV-based blue/green/yellow plate localization."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    candidates = []

    ranges = [
        ("blue", np.array([100, 80, 60]), np.array([140, 255, 255])),
        ("green", np.array([35, 80, 60]), np.array([85, 255, 255])),
        ("yellow", np.array([15, 80, 100]), np.array([40, 255, 255])),
    ]
    ih, iw = img.shape[:2]
    for plate_type, lo, hi in ranges:
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            area = w * h
            if aspect < 2.0 or aspect > 8.0:
                continue
            if area < 1000 or area > iw * ih * 0.5:
                continue
            candidates.append({
                "plate_type": plate_type,
                "confidence": 0.5,
                "box": [x, y, w, h],
                "corners": None,
                "source": "opencv",
            })

    candidates.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
    return candidates[:3]


def greedy_decode_with_scores(logits: torch.Tensor) -> tuple[str, list[dict]]:
    probs = logits.softmax(dim=-1)
    indices = probs.argmax(dim=-1)[0].detach().cpu().tolist()
    confs = probs.max(dim=-1).values[0].detach().cpu().tolist()
    chars = []
    prev = None
    for idx, conf in zip(indices, confs):
        if idx == 0:
            prev = idx
            continue
        if idx == prev:
            continue
        text = decode_indices([idx], collapse_repeats=False)
        chars.append({"text": text, "confidence": round(float(conf), 4)})
        prev = idx
    plate_number = "".join(c["text"] for c in chars)
    return plate_number, chars


class YOLODetector:
    def __init__(self, model_path: str, conf: float, iou: float, imgsz: int, device: str):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        path = Path(model_path)
        if path.suffix == ".onnx":
            self._init_onnx(str(path))
            self.backend = "onnx"
        else:
            self._init_yolo(str(path), device)
            self.backend = "yolo"

    def _init_yolo(self, path: str, device: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("ultralytics not installed: pip install ultralytics")
        self.model = YOLO(path)
        self.device = device

    def _init_onnx(self, path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        if self.backend == "yolo":
            return self._detect_yolo(img_bgr)
        return self._detect_onnx(img_bgr)

    def _detect_yolo(self, img_bgr: np.ndarray) -> list[dict]:
        results = self.model(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            # Pose models attach 4 corner keypoints per detection; use them for warp.
            kpts = None
            if getattr(r, "keypoints", None) is not None and r.keypoints is not None:
                try:
                    kpts = r.keypoints.xy.cpu().numpy()  # [N, K, 2] in original image coords
                except Exception:
                    kpts = None
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().tolist()
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                plate_type = YOLO_CLASS_NAMES[cls_id] if cls_id < len(YOLO_CLASS_NAMES) else "unknown"
                corners = None
                if kpts is not None and i < len(kpts) and kpts[i].shape[0] == 4:
                    corners = [[float(p[0]), float(p[1])] for p in kpts[i]]
                detections.append({
                    "plate_type": plate_type,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "corners": corners,
                })
        return detections

    def _detect_onnx(self, img_bgr: np.ndarray) -> list[dict]:
        """Parse raw YOLOv8 ONNX output.

        YOLOv8 export gives a single tensor shaped [1, 4+nc, N] (channels-first):
        rows 0..3 are cx, cy, w, h (in imgsz space), rows 4..4+nc are per-class scores.
        We transpose to [N, 4+nc], take per-row best class, threshold, then apply NMS.
        """
        ih, iw = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img_rgb, (self.imgsz, self.imgsz))
        inp = img_r.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: inp})[0]  # [1, 4+nc, N]

        pred = np.squeeze(out, axis=0)
        # Ensure shape is [N, 4+nc]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)

        keep = confs >= self.conf
        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        confs = confs[keep]
        if len(boxes_xywh) == 0:
            return []

        scale_x = iw / self.imgsz
        scale_y = ih / self.imgsz

        # cx,cy,w,h (imgsz space) -> xywh top-left in original space, for NMS use xywh ints
        nms_boxes = []
        meta = []
        for (cx, cy, w, h), conf, cls_id in zip(boxes_xywh, confs, cls_ids):
            x = (cx - w / 2) * scale_x
            y = (cy - h / 2) * scale_y
            ww = w * scale_x
            hh = h * scale_y
            nms_boxes.append([float(x), float(y), float(ww), float(hh)])
            meta.append((float(conf), int(cls_id)))

        indices = cv2.dnn.NMSBoxes(nms_boxes, [m[0] for m in meta], self.conf, self.iou)
        detections = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, ww, hh = nms_boxes[i]
                conf, cls_id = meta[i]
                plate_type = YOLO_CLASS_NAMES[cls_id] if cls_id < len(YOLO_CLASS_NAMES) else "unknown"
                detections.append({
                    "plate_type": plate_type,
                    "confidence": round(conf, 4),
                    "box": [x, y, ww, hh],
                    "corners": None,
                })
        return detections


class OCRRecognizer:
    def __init__(self, checkpoint_path: str, device: str, crop_size: tuple[int, int]):
        self.crop_size = crop_size
        dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.device = dev
        checkpoint = torch.load(checkpoint_path, map_location=dev)
        self.model = build_model_from_checkpoint(checkpoint, device=dev)
        self.image_size = get_checkpoint_image_size(checkpoint)

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, list[dict]]:
        h, w = self.image_size
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img_r.astype(np.float32) / 255.0)
        tensor = ((tensor - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return greedy_decode_with_scores(logits)


def process_image(
    img_path: str,
    detector,
    recognizer: OCRRecognizer,
    crop_height: int,
    crop_width: int,
    yolo_available: bool,
) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {"image_path": img_path, "error": "cannot_read", "plates": [], "elapsed_ms": 0.0}

    t0 = time.time()

    if yolo_available:
        raw_dets = detector.detect(img)
    else:
        raw_dets = opencv_detect_plates(img)

    plates = []
    for det in raw_dets:
        t_ocr0 = time.time()
        corners = det.get("corners")
        box = det["box"]

        if corners and len(corners) == 4:
            crop = warp_plate(img, corners, crop_width, crop_height)
        else:
            crop = crop_from_box(img, box, crop_width, crop_height)

        plate_str, char_list = recognizer.recognize(crop)
        elapsed_ocr_ms = (time.time() - t_ocr0) * 1000

        plate_type_det = det.get("plate_type", "unknown")
        plate_type_ocr = infer_plate_type_from_length(plate_str, fallback=plate_type_det)
        rule_result = validate_china_plate(plate_str, plate_type=plate_type_ocr)

        avg_conf = float(np.mean([c["confidence"] for c in char_list])) if char_list else 0.0
        det_conf = det.get("confidence", 0.0)
        combined_conf = round((avg_conf + det_conf) / 2, 4) if det_conf > 0 else round(avg_conf, 4)

        x, y, w, h = box
        plates.append({
            "plate_number": rule_result.text,
            "plate_type": plate_type_ocr,
            "confidence": combined_conf,
            "box": [round(float(v), 2) for v in [x, y, w, h]],
            "corners": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in corners] if corners else None,
            "characters": char_list,
            "valid": rule_result.valid,
            "rule_reason": rule_result.reason,
            "elapsed_ms": round(elapsed_ocr_ms, 3),
        })

    elapsed_ms = (time.time() - t0) * 1000
    return {
        "image_path": str(Path(img_path).resolve()),
        "plates": plates,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def main():
    args = parse_args()

    recognizer = OCRRecognizer(
        checkpoint_path=args.ocr_checkpoint,
        device=args.device,
        crop_size=(args.crop_height, args.crop_width),
    )

    if args.yolo_model:
        detector = YOLODetector(
            model_path=args.yolo_model,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            device=args.device,
        )
        yolo_available = True
    else:
        detector = None
        yolo_available = False
        if not args.quiet:
            print("Warning: no YOLO model provided, using OpenCV HSV fallback.", file=sys.stderr)

    results = []
    for img_path in args.images:
        result = process_image(
            img_path=img_path,
            detector=detector,
            recognizer=recognizer,
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            yolo_available=yolo_available,
        )
        results.append(result)
        if args.vis_dir:
            vis_path = Path(args.vis_dir) / (Path(img_path).stem + "_vis.png")
            render_visualization(img_path, result, vis_path)
            if not args.quiet:
                print(f"  -> visualization: {vis_path}")
        if not args.quiet:
            n = len(result["plates"])
            plates_str = ", ".join(p["plate_number"] for p in result["plates"])
            print(f"{img_path}: {n} plate(s) [{plates_str}] ({result['elapsed_ms']:.1f} ms)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"Saved results to {args.output}")
    else:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
