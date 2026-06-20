#!/usr/bin/env python
"""Train CRNN-lite for cropped plate OCR."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from plate_course.chars import CHARSET, greedy_decode
from plate_course.dataset import PlateOCRDataset, collate_plate_batch
from plate_course.metrics import recognition_metrics
from plate_course.model import build_recognizer, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/processed/ocr_5color/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/processed/ocr_5color/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/models/crnn_lite")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--model", default="crnn_lite", choices=["crnn_lite", "lprnet_lite"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1, help="BiGRU layers for CRNN.")
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--amp", action="store_true", help="Mixed-precision (faster on 4090).")
    parser.add_argument("--augment", action="store_true", help="Train-time data augmentation.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(manifest, image_size, batch_size, num_workers, shuffle, max_samples=None, augment=False):
    dataset = PlateOCRDataset(manifest, image_size=image_size, max_samples=max_samples, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_plate_batch,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    predictions: list[str] = []
    targets: list[str] = []
    start = time.time()
    use_amp = scaler is not None and device.type == "cuda"

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        flat_targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        labels = batch["labels"]

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2).contiguous().float()
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=logits.size(1),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_items += images.size(0)
        predictions.extend(greedy_decode(logits.float()))
        targets.extend(labels)

    elapsed = time.time() - start
    metrics = recognition_metrics(predictions, targets)
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["elapsed_s"] = elapsed
    metrics["fps"] = total_items / max(elapsed, 1e-6)
    return metrics, predictions, targets


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "charset": CHARSET,
        "args": vars(args),
        "config": {
            "model_name": args.model,
            "num_classes": len(CHARSET),
            "charset": CHARSET,
        },
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(
        args.train_manifest,
        image_size,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        max_samples=args.max_train_samples,
        augment=args.augment,
    )
    val_loader = make_loader(
        args.val_manifest,
        image_size,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        max_samples=args.max_val_samples,
    )

    model = build_recognizer(
        args.model, num_classes=len(CHARSET), hidden_size=args.hidden_size, num_layers=args.num_layers
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    config = {
        "model": args.model,
        "model_name": args.model,
        "num_classes": len(CHARSET),
        "charset": CHARSET,
        "image_size": {"height": args.image_height, "width": args.image_width},
        "normalization": "RGB float32, resized to 160x48, value=(x/255-0.5)/0.5",
        "decode": "CTC greedy decode, blank index 0",
        "params": count_parameters(model),
        "single_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "args": vars(args),
    }
    (output_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "train_log.csv"
    best_score = (-1.0, -1.0)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_plate_accuracy",
                "train_character_accuracy",
                "val_loss",
                "val_plate_accuracy",
                "val_character_accuracy",
                "val_avg_edit_distance",
                "val_fps",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
            val_metrics, val_preds, val_targets = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_plate_accuracy": train_metrics["plate_accuracy"],
                "train_character_accuracy": train_metrics["character_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_plate_accuracy": val_metrics["plate_accuracy"],
                "val_character_accuracy": val_metrics["character_accuracy"],
                "val_avg_edit_distance": val_metrics["avg_edit_distance"],
                "val_fps": val_metrics["fps"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:03d} lr={lr:.2e} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_plate_acc={val_metrics['plate_accuracy']:.4f} "
                f"val_char_acc={val_metrics['character_accuracy']:.4f}"
            )

            save_checkpoint(last_path, model, optimizer, epoch, val_metrics, args)
            score = (val_metrics["plate_accuracy"], val_metrics["character_accuracy"])
            if score > best_score:
                best_score = score
                save_checkpoint(best_path, model, optimizer, epoch, val_metrics, args)
                examples = [
                    {"target": t, "prediction": p, "correct": t == p}
                    for p, t in zip(val_preds[:80], val_targets[:80])
                ]
                (output_dir / "val_examples.json").write_text(
                    json.dumps(examples, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
