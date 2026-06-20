#!/usr/bin/env python
"""Evaluate CRNN-lite checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from plate_course.chars import CHARSET, greedy_decode
from plate_course.dataset import PlateOCRDataset, collate_plate_batch
from plate_course.metrics import recognition_metrics
from plate_course.runtime import build_model_from_checkpoint, get_checkpoint_image_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/models/crnn_lite/best.pt")
    parser.add_argument("--manifest", default="data/processed/ocr_5color/manifests/test.jsonl")
    parser.add_argument("--output-json", default="outputs/models/crnn_lite/test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def grouped_metrics(predictions: list[str], targets: list[str], records: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, list[str]]] = {}
    for pred, target, record in zip(predictions, targets, records):
        value = str(record.get(key, "unknown"))
        groups.setdefault(value, {"pred": [], "target": []})
        groups[value]["pred"].append(pred)
        groups[value]["target"].append(target)
    output = {}
    for value, items in sorted(groups.items()):
        metrics = recognition_metrics(items["pred"], items["target"])
        metrics["samples"] = len(items["target"])
        output[value] = metrics
    return output


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = get_checkpoint_image_size(checkpoint)

    dataset = PlateOCRDataset(args.manifest, image_size=image_size, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_plate_batch,
    )
    model = build_model_from_checkpoint(checkpoint, device=device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    predictions: list[str] = []
    targets: list[str] = []
    records: list[dict] = []
    losses = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            flat_targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2).contiguous()
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
            losses.append(float(loss.item()) * images.size(0))
            predictions.extend(greedy_decode(logits))
            targets.extend(batch["labels"])
            records.extend(batch["records"])

    elapsed = time.time() - start
    metrics = recognition_metrics(predictions, targets)
    metrics.update(
        {
            "loss": sum(losses) / max(len(dataset), 1),
            "samples": len(dataset),
            "elapsed_ms_per_image": elapsed * 1000 / max(len(dataset), 1),
            "fps": len(dataset) / max(elapsed, 1e-6),
        }
    )
    examples = [
        {"target": t, "prediction": p, "correct": p == t}
        for p, t in zip(predictions[:120], targets[:120])
    ]
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model": checkpoint.get("args", {}).get("model", "crnn_lite"),
        "metrics": metrics,
        "by_plate_type": grouped_metrics(predictions, targets, records, "plate_type"),
        "by_mode": grouped_metrics(predictions, targets, records, "mode"),
        "by_source": grouped_metrics(predictions, targets, records, "source"),
        "examples": examples,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
