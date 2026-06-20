#!/usr/bin/env python
"""Batch evaluation script: OCR metrics on a test manifest, optionally YOLO detection metrics.

Usage (OCR only, using pre-cropped plate images):
    python scripts/evaluate.py \
        --ocr-checkpoint outputs/models/crnn_lite_mixed_current_e40/best.pt \
        --test-manifest data/processed/mixed_plate_ocr_current/manifests/test.jsonl \
        --output-json outputs/metrics/eval_ocr.json

Usage (full pipeline with YOLO, on scene images from manifest):
    python scripts/evaluate.py \
        --ocr-checkpoint outputs/models/crnn_lite_mixed_current_e40/best.pt \
        --yolo-model outputs/yolo/best.pt \
        --test-manifest data/processed/ccpd_green_subset/manifests/test.jsonl \
        --mode pipeline \
        --output-json outputs/metrics/eval_pipeline.json
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
from torch.utils.data import DataLoader

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from plate_course.chars import greedy_decode
from plate_course.dataset import PlateOCRDataset, collate_plate_batch
from plate_course.metrics import edit_distance, recognition_metrics
from plate_course.postprocess import validate_china_plate
from plate_course.runtime import build_model_from_checkpoint, get_checkpoint_image_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument(
        "--mode",
        choices=["ocr", "pipeline"],
        default="ocr",
        help="ocr: evaluate on pre-cropped plate images. pipeline: YOLO detect + OCR.",
    )
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", default="outputs/metrics/eval_results.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for detection TP.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


def grouped_metrics(preds: list[str], targets: list[str], records: list[dict], key: str) -> dict:
    groups: dict[str, dict[str, list]] = {}
    for p, t, r in zip(preds, targets, records):
        v = str(r.get(key, "unknown"))
        groups.setdefault(v, {"pred": [], "target": []})
        groups[v]["pred"].append(p)
        groups[v]["target"].append(t)
    out = {}
    for v, items in sorted(groups.items()):
        m = recognition_metrics(items["pred"], items["target"])
        m["samples"] = len(items["target"])
        out[v] = m
    return out


def eval_ocr_mode(args, device: torch.device) -> dict:
    checkpoint = torch.load(args.ocr_checkpoint, map_location=device)
    image_size = get_checkpoint_image_size(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device)

    dataset = PlateOCRDataset(args.test_manifest, image_size=image_size, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_plate_batch,
    )

    predictions: list[str] = []
    targets: list[str] = []
    records: list[dict] = []
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            logits = model(images)
            predictions.extend(greedy_decode(logits))
            targets.extend(batch["labels"])
            records.extend(batch["records"])

    elapsed = time.time() - start
    metrics = recognition_metrics(predictions, targets)
    metrics.update({
        "samples": len(dataset),
        "elapsed_ms_per_image": elapsed * 1000 / max(len(dataset), 1),
        "fps": len(dataset) / max(elapsed, 1e-6),
    })

    by_type = grouped_metrics(predictions, targets, records, "plate_type")
    by_source = grouped_metrics(predictions, targets, records, "source")

    examples_correct = [
        {"target": t, "prediction": p, "correct": t == p}
        for p, t in zip(predictions, targets) if p == t
    ][:20]
    examples_wrong = [
        {"target": t, "prediction": p, "correct": t == p}
        for p, t in zip(predictions, targets) if p != t
    ][:20]

    return {
        "mode": "ocr",
        "checkpoint": str(Path(args.ocr_checkpoint).resolve()),
        "manifest": str(Path(args.test_manifest).resolve()),
        "metrics": metrics,
        "by_plate_type": by_type,
        "by_source": by_source,
        "success_examples": examples_correct,
        "failure_examples": examples_wrong,
    }


def eval_pipeline_mode(args, device: torch.device) -> dict:
    from infer import OCRRecognizer, YOLODetector, warp_plate, crop_from_box  # noqa: F401

    if not args.yolo_model:
        raise ValueError("--yolo-model is required in pipeline mode")

    recognizer = OCRRecognizer(
        checkpoint_path=args.ocr_checkpoint,
        device=str(device),
        crop_size=(48, 160),
    )
    detector = YOLODetector(
        model_path=args.yolo_model,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        device=str(device),
    )

    records = read_jsonl(args.test_manifest)
    if args.max_samples:
        records = records[:args.max_samples]

    tp_det = 0
    fp_det = 0
    fn_det = 0
    ocr_preds: list[str] = []
    ocr_targets: list[str] = []
    ocr_records: list[dict] = []

    for rec in records:
        scene_path = rec.get("scene_path")
        gt_box = rec.get("box_xywh")
        gt_plate = rec.get("plate_number", "")
        if not scene_path or not Path(scene_path).is_file():
            fn_det += 1
            continue

        img = cv2.imread(scene_path)
        if img is None:
            fn_det += 1
            continue

        dets = detector.detect(img)
        best_det = None
        best_iou = 0.0
        if gt_box:
            for det in dets:
                iou = compute_iou(det["box"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

        if best_det is not None and best_iou >= args.iou_threshold:
            tp_det += 1
            corners = best_det.get("corners")
            if corners and len(corners) == 4:
                crop = warp_plate(img, corners, 160, 48)
            else:
                crop = crop_from_box(img, best_det["box"], 160, 48)
            plate_str, _ = recognizer.recognize(crop)
            ocr_preds.append(plate_str)
            ocr_targets.append(gt_plate)
            ocr_records.append(rec)
        elif dets:
            fp_det += 1
        else:
            fn_det += 1

    precision = tp_det / max(tp_det + fp_det, 1)
    recall = tp_det / max(tp_det + fn_det, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    det_metrics = {
        "TP": tp_det,
        "FP": fp_det,
        "FN": fn_det,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "iou_threshold": args.iou_threshold,
    }

    ocr_metrics = recognition_metrics(ocr_preds, ocr_targets) if ocr_preds else {}

    return {
        "mode": "pipeline",
        "checkpoint_ocr": str(Path(args.ocr_checkpoint).resolve()),
        "checkpoint_yolo": str(Path(args.yolo_model).resolve()),
        "manifest": str(Path(args.test_manifest).resolve()),
        "detection_metrics": det_metrics,
        "ocr_metrics": ocr_metrics,
        "by_plate_type": grouped_metrics(ocr_preds, ocr_targets, ocr_records, "plate_type"),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.mode == "ocr":
        result = eval_ocr_mode(args, device)
    else:
        result = eval_pipeline_mode(args, device)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    m = result.get("metrics") or result.get("ocr_metrics") or {}
    print(json.dumps(m, ensure_ascii=False, indent=2))
    if "detection_metrics" in result:
        print("Detection:", json.dumps(result["detection_metrics"], ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
