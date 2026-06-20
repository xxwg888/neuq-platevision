#!/usr/bin/env python
"""Train YOLOv8n for Chinese license plate detection.

Usage:
    python scripts/train_yolo.py \
        --data datasets/yolo/plate_detect.yaml \
        --output-dir outputs/yolo \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --device 0

After training, best.pt is copied to outputs/yolo/best.pt.
Run with --export-onnx to also export best.onnx.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/yolo/plate_detect.yaml")
    parser.add_argument("--output-dir", default="outputs/yolo")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model: yolov8n.pt or yolov5n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id or 'cpu'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--export-onnx", action="store_true", help="Export best.onnx after training")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        weight_decay=args.weight_decay,
        project=str(output_dir.resolve()),
        name="train",
        exist_ok=True,
        val=True,
        plots=True,
        save=True,
    )

    # Resolve the actual save dir from the trainer (robust to ultralytics runs_dir settings).
    save_dir = None
    try:
        save_dir = Path(model.trainer.save_dir)
    except Exception:
        save_dir = output_dir / "train"

    best_src = save_dir / "weights" / "best.pt"
    best_dst = output_dir / "best.pt"
    if best_src.is_file():
        shutil.copy2(best_src, best_dst)
        print(f"Copied best weights ({best_src}) -> {best_dst}")
    else:
        print(f"Warning: best.pt not found at {best_src}")

    metrics_path = save_dir / "results.json"
    summary = {"args": vars(args), "save_dir": str(save_dir)}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            summary["results"] = json.load(f)

    try:
        val_metrics = results.results_dict if hasattr(results, "results_dict") else {}
        summary["final_metrics"] = val_metrics
    except Exception:
        pass

    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.export_onnx and best_dst.is_file():
        export_model = YOLO(str(best_dst))
        export_path = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=True, opset=17)
        onnx_dst = output_dir / "best.onnx"
        if export_path and Path(export_path).is_file():
            if Path(export_path).resolve() != onnx_dst.resolve():
                shutil.copy2(export_path, onnx_dst)
            print(f"Exported ONNX -> {onnx_dst}")

    print("Done. Best checkpoint:", best_dst)


if __name__ == "__main__":
    main()
