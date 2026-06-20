#!/usr/bin/env python
"""Run CRNN-lite prediction on cropped plate images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from plate_course.chars import CHARSET, decode_indices, greedy_decode
from plate_course.dataset import preprocess_plate_image
from plate_course.postprocess import infer_plate_type_from_length, validate_china_plate
from plate_course.runtime import build_model_from_checkpoint, get_checkpoint_image_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="outputs/models/crnn_lite/best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plate-type", default="unknown")
    return parser.parse_args()


def greedy_decode_char_scores(logits: torch.Tensor) -> list[dict[str, float]]:
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
        chars.append({"text": decode_indices([idx], collapse_repeats=False), "confidence": round(float(conf), 4)})
        prev = idx
    return chars


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = get_checkpoint_image_size(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device)

    results = []
    with torch.no_grad():
        for image_path in args.images:
            tensor = preprocess_plate_image(image_path, image_size=image_size).unsqueeze(0).to(device)
            start = time.time()
            logits = model(tensor)
            elapsed_ms = (time.time() - start) * 1000
            pred = greedy_decode(logits)[0]
            prob = logits.softmax(dim=-1).max(dim=-1).values.mean().item()
            plate_type = infer_plate_type_from_length(pred, fallback=args.plate_type)
            rule_result = validate_china_plate(pred, plate_type=plate_type if plate_type != "unknown" else None)
            results.append(
                {
                    "image_path": str(Path(image_path).resolve()),
                    "plate_number": rule_result.text,
                    "plate_type": plate_type,
                    "confidence": round(float(prob), 4),
                    "box": None,
                    "corners": None,
                    "characters": greedy_decode_char_scores(logits),
                    "valid": rule_result.valid,
                    "rule_reason": rule_result.reason,
                    "elapsed_ms": round(elapsed_ms, 3),
                }
            )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
