"""Dataset helpers for plate OCR manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .chars import encode_label


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def augment_plate_bgr(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Light photometric + geometric augmentation for real-plate OCR robustness.

    Kept mild so characters stay legible (CTC needs readable glyphs). Applied on the
    BGR crop before resize/normalize. Train-time only.
    """
    h, w = img.shape[:2]

    if rng.rand() < 0.7:  # small affine: rotation + scale + translation
        angle = rng.uniform(-6, 6)
        scale = rng.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += rng.uniform(-0.05, 0.05) * w
        M[1, 2] += rng.uniform(-0.08, 0.08) * h
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.rand() < 0.4:  # mild perspective (residual viewing angle)
        d = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, src.shape).astype(np.float32) * np.float32([w, h])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)

    if rng.rand() < 0.7:  # brightness / contrast
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))

    if rng.rand() < 0.4:  # hue / saturation jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 7)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    r = rng.rand()  # blur or noise
    if r < 0.25:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.4:
        noise = rng.randn(h, w, 3).astype(np.float32) * rng.uniform(4, 12)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def preprocess_plate_image(path: str | Path, image_size: tuple[int, int] = (48, 160),
                           augment: bool = False, rng: np.random.RandomState | None = None) -> torch.Tensor:
    """Load a plate crop and convert it to a normalized RGB tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if augment:
        img = augment_plate_bgr(img, rng if rng is not None else np.random.RandomState())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = image_size
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


class PlateOCRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: tuple[int, int] = (48, 160),
        max_samples: int | None = None,
        augment: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.records = read_jsonl(self.manifest_path)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        rng = np.random.RandomState() if self.augment else None
        image = preprocess_plate_image(record["image_path"], self.image_size,
                                       augment=self.augment, rng=rng)
        label = record["plate_number"]
        target = torch.tensor(encode_label(label), dtype=torch.long)
        return {
            "image": image,
            "target": target,
            "target_length": len(target),
            "label": label,
            "record": record,
        }


def collate_plate_batch(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = torch.cat([item["target"] for item in batch], dim=0)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    labels = [item["label"] for item in batch]
    records = [item["record"] for item in batch]
    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "labels": labels,
        "records": records,
    }

