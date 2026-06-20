#!/usr/bin/env python
"""Create a compact visualization of plate OCR predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from plate_course.chars import CHARSET, greedy_decode
from plate_course.dataset import PlateOCRDataset
from plate_course.model import CRNNLite


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/models/crnn_lite_mixed_current_e40/best.pt")
    parser.add_argument("--manifest", default="data/processed/mixed_plate_ocr_current/manifests/test.jsonl")
    parser.add_argument("--output", default="outputs/visualizations/current_prediction_examples.jpg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--correct", type=int, default=8)
    parser.add_argument("--wrong", type=int, default=8)
    return parser.parse_args()


def get_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def run_predictions(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})
    hidden_size = int(model_args.get("hidden_size", 128))
    image_size = (int(model_args.get("image_height", 48)), int(model_args.get("image_width", 160)))
    dataset = PlateOCRDataset(args.manifest, image_size=image_size, max_samples=args.max_samples)
    model = CRNNLite(num_classes=len(CHARSET), hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            image = item["image"].unsqueeze(0).to(device)
            logits = model(image)
            pred = greedy_decode(logits)[0]
            conf = logits.softmax(dim=-1).max(dim=-1).values.mean().item()
            record = item["record"]
            results.append(
                {
                    "image_path": record["image_path"],
                    "target": item["label"],
                    "prediction": pred,
                    "correct": pred == item["label"],
                    "confidence": conf,
                    "plate_type": record.get("plate_type", "unknown"),
                    "mode": record.get("mode", "unknown"),
                    "source": record.get("source", "unknown"),
                }
            )
    return results


def make_canvas(results, output: Path, correct_n: int, wrong_n: int):
    correct = [r for r in results if r["correct"]][:correct_n]
    wrong = [r for r in results if not r["correct"]][:wrong_n]
    selected = correct + wrong
    cell_w, cell_h = 360, 132
    cols = 2
    rows = max(1, (len(selected) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(18)
    text_font = get_font(15)
    small_font = get_font(13)

    for idx, record in enumerate(selected):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        bg = (232, 247, 238) if record["correct"] else (252, 236, 236)
        draw.rectangle((x + 6, y + 6, x + cell_w - 6, y + cell_h - 6), fill=bg, outline=(210, 215, 222))
        img = Image.open(record["image_path"]).convert("RGB")
        img.thumbnail((155, 58))
        canvas.paste(img, (x + 14, y + 18))
        mark = "OK" if record["correct"] else "ERR"
        color = (18, 128, 74) if record["correct"] else (190, 45, 45)
        draw.text((x + 182, y + 14), mark, font=title_font, fill=color)
        draw.text((x + 182, y + 40), f"GT: {record['target']}", font=text_font, fill=(25, 30, 40))
        draw.text((x + 182, y + 64), f"Pred: {record['prediction']}", font=text_font, fill=(25, 30, 40))
        draw.text(
            (x + 14, y + 86),
            f"{record['source']} | {record['plate_type']} | {record['mode']} | conf {record['confidence']:.2f}",
            font=small_font,
            fill=(70, 78, 90),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main():
    args = parse_args()
    results = run_predictions(args)
    output = Path(args.output)
    make_canvas(results, output, args.correct, args.wrong)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    correct = sum(1 for r in results if r["correct"])
    print(f"saved: {output}")
    print(f"json: {json_path}")
    print(f"exact_correct: {correct}/{len(results)}")


if __name__ == "__main__":
    main()

