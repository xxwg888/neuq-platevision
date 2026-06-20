#!/usr/bin/env python
"""Train YOLOv8n-pose for plate detection + 4-corner keypoint regression.

The 4 keypoints (TL, TR, BR, BL) enable perspective correction at inference so the
OCR input matches the warped crops used in OCR training.

Usage:
    python scripts/train_yolo_pose.py \
        --data datasets/yolo_pose/plate_pose.yaml \
        --output-dir outputs/yolo_pose --epochs 120 --device 1 --export-onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/yolo_pose/plate_pose.yaml")
    p.add_argument("--output-dir", default="outputs/yolo_pose")
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--export-onnx", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install ultralytics")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        lr0=args.lr0,
        project=str(out),
        name="train",
        exist_ok=True,
        val=True,
        plots=True,
        save=True,
    )

    best_src = out / "train" / "weights" / "best.pt"
    best_dst = out / "best.pt"
    if best_src.is_file():
        shutil.copy2(best_src, best_dst)
        print(f"Copied best weights -> {best_dst}")

    try:
        summary = {"args": vars(args), "final_metrics": results.results_dict if hasattr(results, "results_dict") else {}}
        (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    if args.export_onnx and best_dst.is_file():
        export_model = YOLO(str(best_dst))
        export_path = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=True, opset=17)
        onnx_dst = out / "best.onnx"
        if export_path and Path(export_path).is_file():
            if Path(export_path).resolve() != onnx_dst.resolve():
                shutil.copy2(export_path, onnx_dst)
            print(f"Exported ONNX -> {onnx_dst}")

    print("Done. Best:", best_dst)


if __name__ == "__main__":
    main()
