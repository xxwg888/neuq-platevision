#!/usr/bin/env python
"""Export CRNN-lite or LPRNet-lite checkpoint to ONNX, or export a YOLO checkpoint.

Usage:
    # Export OCR model to ocr_best.onnx:
    python scripts/export_onnx.py \
        --checkpoint outputs/models/crnn_lite_mixed_current_e40/best.pt \
        --output outputs/models/ocr_best.onnx

    # Export YOLO to ONNX (uses ultralytics):
    python scripts/export_onnx.py \
        --yolo-checkpoint outputs/yolo/best.pt \
        --output outputs/yolo/best.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from plate_course.chars import CHARSET
from plate_course.model import build_recognizer
from plate_course.runtime import get_checkpoint_args, get_checkpoint_model_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="OCR .pt checkpoint path.")
    parser.add_argument("--yolo-checkpoint", default=None, help="YOLO .pt checkpoint (uses ultralytics).")
    parser.add_argument("--output", default=None, help="Output .onnx path (auto-inferred if not set).")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    return parser.parse_args()


def export_ocr(checkpoint_path: str, output_path: str | None, opset: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args_dict = get_checkpoint_args(checkpoint)
    model_name = get_checkpoint_model_name(checkpoint)
    hidden_size = int(args_dict.get("hidden_size", 128))
    num_layers = int(args_dict.get("num_layers", 1))
    image_h = int(args_dict.get("image_height", 48))
    image_w = int(args_dict.get("image_width", 160))

    model = build_recognizer(model_name, num_classes=len(CHARSET), hidden_size=hidden_size,
                             num_layers=num_layers)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if output_path is None:
        stem = Path(checkpoint_path).stem
        output_path = str(Path(checkpoint_path).parent / f"{stem}.onnx")
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_h, image_w)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch", 1: "time"}},
        opset_version=opset,
    )

    meta = {
        "model": model_name,
        "onnx_path": output_path,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": ["N", 3, image_h, image_w],
        "input_color": "RGB",
        "normalization": f"float32, resize to W={image_w} H={image_h}, value=(x/255-0.5)/0.5",
        "output_shape": ["N", "T", len(CHARSET)],
        "num_classes": len(CHARSET),
        "charset": CHARSET,
        "blank_index": 0,
        "decode": "CTC greedy decode. blank index is 0.",
        "params": sum(p.numel() for p in model.parameters()),
    }
    meta_path = Path(output_path).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported OCR ONNX -> {output_path}")
    print(f"Meta           -> {meta_path}")
    return output_path


def export_yolo(checkpoint_path: str, output_path: str | None, imgsz: int):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install ultralytics")
    model = YOLO(checkpoint_path)
    export_result = model.export(format="onnx", imgsz=imgsz, dynamic=True, opset=17)
    src = Path(str(export_result))
    if output_path:
        import shutil
        shutil.copy2(src, output_path)
        print(f"Exported YOLO ONNX -> {output_path}")
    else:
        print(f"Exported YOLO ONNX -> {src}")


def main():
    args = parse_args()
    if args.checkpoint:
        export_ocr(args.checkpoint, args.output, args.opset)
    elif args.yolo_checkpoint:
        export_yolo(args.yolo_checkpoint, args.output, args.yolo_imgsz)
    else:
        raise SystemExit("Provide --checkpoint (OCR) or --yolo-checkpoint (YOLO).")


if __name__ == "__main__":
    main()
