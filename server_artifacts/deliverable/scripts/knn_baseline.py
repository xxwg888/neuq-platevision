#!/usr/bin/env python
"""KNN baseline for license plate character recognition.

Pipeline:
  1. Load plate crop (already rectified, 96x320 or 48x160)
  2. Grayscale → Otsu threshold
  3. Character segmentation via horizontal projection (vertical projection within each char row)
  4. Each character patch → resize to 20x40 → HOG features
  5. KNN train/predict
  6. Concatenate character predictions → full plate string

Usage (train + evaluate):
    python scripts/knn_baseline.py \
        --train-manifest data/processed/mixed_plate_ocr_current/manifests/train.jsonl \
        --test-manifest data/processed/mixed_plate_ocr_current/manifests/test.jsonl \
        --output-json outputs/metrics/knn_baseline.json \
        --save-model outputs/models/knn/knn_baseline.pkl

Usage (evaluate only):
    python scripts/knn_baseline.py \
        --test-manifest data/processed/mixed_plate_ocr_current/manifests/test.jsonl \
        --load-model outputs/models/knn/knn_baseline.pkl \
        --output-json outputs/metrics/knn_baseline.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from plate_course.metrics import recognition_metrics

CHAR_H = 40
CHAR_W = 24  # (W-blockW)%blockStride == 0 must hold: (24-16)%8==0, (40-16)%8==0
HOG_WIN = (CHAR_H, CHAR_W)
HOG_CELL = (8, 8)
HOG_BLOCK = (2, 2)
HOG_BINS = 9

PROVINCES = list("京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳")
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("I", "O")]
DIGITS = list("0123456789")
SPECIALS = list("警学挂领使临")
ALL_CHARS = PROVINCES + LETTERS + DIGITS + SPECIALS
CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", default=None)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--output-json", default="outputs/metrics/knn_baseline.json")
    p.add_argument("--save-model", default=None)
    p.add_argument("--load-model", default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--plate-height", type=int, default=48)
    p.add_argument("--plate-width", type=int, default=160)
    p.add_argument("--blue-only", action="store_true", help="Only evaluate on blue plates (7-char).")
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_hog():
    return cv2.HOGDescriptor(
        HOG_WIN[::-1],
        (HOG_BLOCK[0] * HOG_CELL[0], HOG_BLOCK[1] * HOG_CELL[1]),
        (HOG_CELL[0], HOG_CELL[1]),
        (HOG_CELL[0], HOG_CELL[1]),
        HOG_BINS,
    )


HOG = make_hog()


def hog_features(patch: np.ndarray) -> np.ndarray:
    patch_r = cv2.resize(patch, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
    return HOG.compute(patch_r).flatten()


def load_plate(path: str | Path, plate_w: int, plate_h: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (plate_w, plate_h), interpolation=cv2.INTER_AREA)


def binarize(img_bgr: np.ndarray) -> np.ndarray:
    """Border-trimmed, denoised binary plate with white characters on black.

    Real warped crops carry a coloured frame and rivets that wreck pure projection
    segmentation, so we trim a small margin, Otsu-threshold, orient characters to
    white, and morphologically open to drop speckle/rivets.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    my, mx = int(round(h * 0.10)), int(round(w * 0.02))  # trim frame
    gray = gray[my:h - my, mx:w - mx] if (h - 2 * my > 8 and w - 2 * mx > 8) else gray
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = cv2.bitwise_not(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return bw


def _smooth1d(a: np.ndarray, k: int = 5) -> np.ndarray:
    return np.convolve(a.astype(np.float32), np.ones(k, np.float32) / k, mode="same")


def segment_runs_to_n(bw: np.ndarray, n_chars: int) -> list[tuple[int, int]]:
    """Like segment_to_n but returns the (x_start, x_end) column runs instead of
    patches, so callers can report per-character bounding boxes. Single source of
    truth for the force-to-N segmentation used by every baseline + the demo."""
    h, w = bw.shape
    proj = _smooth1d(bw.sum(axis=0), 5)
    thr = proj.max() * 0.10
    runs: list[list[int]] = []
    x_start = 0
    in_char = False
    for x in range(w):
        if not in_char and proj[x] > thr:
            x_start = x
            in_char = True
        elif in_char and proj[x] <= thr:
            if x - x_start >= 2:
                runs.append([x_start, x])
            in_char = False
    if in_char and w - x_start >= 2:
        runs.append([x_start, w])
    if not runs:
        return []

    while len(runs) > n_chars:  # merge smallest-gap neighbours
        gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
        _, i = min(gaps)
        runs[i] = [runs[i][0], runs[i + 1][1]]
        del runs[i + 1]
    while len(runs) < n_chars:  # split the widest run
        widths = [(runs[i][1] - runs[i][0], i) for i in range(len(runs))]
        _, i = max(widths)
        s, e = runs[i]
        if e - s < 2:
            break
        mid = (s + e) // 2
        runs[i] = [s, mid]
        runs.insert(i + 1, [mid, e])

    return [(s, e) for s, e in runs if e > s]


def segment_to_n(bw: np.ndarray, n_chars: int) -> list[np.ndarray]:
    """Force exactly n_chars patches from a binary plate (see segment_runs_to_n)."""
    return [bw[:, s:e] for s, e in segment_runs_to_n(bw, n_chars)]


def segment_runs(bw: np.ndarray, n_chars: int | None = None) -> list[tuple[int, int]]:
    """Unified (x_start, x_end) column runs matching segment_characters_projection.

    If n_chars is known -> force-to-N runs; else free-running projection. Mirrors the
    exact thresholds used by the baselines so per-char bboxes line up with the patches.
    """
    if n_chars:
        runs = segment_runs_to_n(bw, n_chars)
        if runs:
            return runs
    w = bw.shape[1]
    col_proj = _smooth1d(bw.sum(axis=0), 5)
    threshold = col_proj.max() * 0.10
    in_char = False
    runs = []
    x_start = 0
    for x in range(w):
        if not in_char and col_proj[x] > threshold:
            x_start = x
            in_char = True
        elif in_char and col_proj[x] <= threshold:
            if x - x_start >= 3:
                runs.append((x_start, x))
            in_char = False
    if in_char and w - x_start >= 3:
        runs.append((x_start, w))
    return runs


def segment_characters_projection(bw: np.ndarray, n_chars: int | None = None) -> list[np.ndarray]:
    if n_chars:
        segs = segment_to_n(bw, n_chars)
        if segs:
            return segs
    # n unknown (or failed): free-running projection.
    w = bw.shape[1]
    col_proj = _smooth1d(bw.sum(axis=0), 5)
    threshold = col_proj.max() * 0.10
    in_char = False
    segments = []
    x_start = 0
    for x in range(w):
        if not in_char and col_proj[x] > threshold:
            x_start = x
            in_char = True
        elif in_char and col_proj[x] <= threshold:
            seg = bw[:, x_start:x]
            if seg.shape[1] >= 3:
                segments.append(seg)
            in_char = False
    if in_char:
        seg = bw[:, x_start:]
        if seg.shape[1] >= 3:
            segments.append(seg)
    return segments


def _merge_or_split(segs: list, bw: np.ndarray, n_chars: int, col_proj: np.ndarray, threshold: float) -> list:
    if len(segs) > n_chars:
        avg_w = bw.shape[1] / n_chars
        merged = []
        current = []
        current_w = 0
        for s in segs:
            if current_w + s.shape[1] <= avg_w * 1.5:
                current.append(s)
                current_w += s.shape[1]
            else:
                if current:
                    merged.append(np.concatenate(current, axis=1))
                current = [s]
                current_w = s.shape[1]
        if current:
            merged.append(np.concatenate(current, axis=1))
        segs = merged
    return segs[:n_chars] if len(segs) >= n_chars else segs


def segment_equal_split(bw: np.ndarray, n_chars: int) -> list[np.ndarray]:
    w = bw.shape[1]
    char_w = w // n_chars
    segs = []
    for i in range(n_chars):
        x0 = i * char_w
        x1 = x0 + char_w if i < n_chars - 1 else w
        segs.append(bw[:, x0:x1])
    return segs


def extract_plate_char_features(img_bgr: np.ndarray, n_chars: int | None, plate_w: int, plate_h: int) -> list[np.ndarray] | None:
    bw = binarize(img_bgr)
    segs = segment_characters_projection(bw, n_chars)
    if not segs and n_chars:
        segs = segment_equal_split(bw, n_chars)
    if not segs:
        return None
    return [hog_features(s) for s in segs]


def build_character_dataset(
    records: list[dict], plate_w: int, plate_h: int, max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = []
    y = []
    labels = []
    for rec in records[:max_samples] if max_samples else records:
        img_path = rec.get("image_path")
        plate_number = rec.get("plate_number", "")
        if not img_path or not plate_number:
            continue
        if not Path(img_path).is_file():
            continue
        img = load_plate(img_path, plate_w, plate_h)
        if img is None:
            continue
        n_chars = len(plate_number)
        feats = extract_plate_char_features(img, n_chars, plate_w, plate_h)
        if feats is None or len(feats) != n_chars:
            continue
        for feat, char in zip(feats, plate_number):
            if char not in CHAR_TO_IDX:
                continue
            X.append(feat)
            y.append(CHAR_TO_IDX[char])
            labels.append(char)
    if not X:
        return np.empty((0,)), np.empty((0,)), []
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels


def train_knn(X: np.ndarray, y: np.ndarray, k: int) -> cv2.ml.KNearest:
    knn = cv2.ml.KNearest_create()
    knn.train(X.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.float32))
    knn.setDefaultK(k)
    return knn


def predict_plate(knn: cv2.ml.KNearest, img_bgr: np.ndarray, n_chars: int | None, plate_w: int, plate_h: int, k: int) -> str:
    feats = extract_plate_char_features(img_bgr, n_chars, plate_w, plate_h)
    if not feats:
        return ""
    result = []
    for feat in feats:
        feat_arr = np.array([feat], dtype=np.float32)
        _, results, _, _ = knn.findNearest(feat_arr, k=k)
        idx = int(results[0][0])
        result.append(IDX_TO_CHAR.get(idx, "?"))
    return "".join(result)


def main():
    args = parse_args()

    knn = None
    if args.load_model:
        knn = cv2.ml.KNearest_load(args.load_model)
        knn.setDefaultK(args.k)
        print(f"Loaded KNN from {args.load_model}")
    elif args.train_manifest:
        print("Building character training dataset...")
        train_records = read_jsonl(args.train_manifest)
        if args.blue_only:
            train_records = [r for r in train_records if r.get("plate_type") == "blue"]
        X_train, y_train, _ = build_character_dataset(
            train_records, args.plate_width, args.plate_height, args.max_train_samples
        )
        if len(X_train) == 0:
            raise SystemExit("No training samples could be built. Check image paths in manifest.")
        print(f"  Training samples: {len(X_train)} characters from {len(train_records)} plates")
        print("Training KNN...")
        knn = train_knn(X_train, y_train, args.k)
        if args.save_model:
            Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
            # cv2 ml objects are not picklable; use native FileStorage save.
            save_path = args.save_model
            if not save_path.endswith((".xml", ".yml", ".yaml")):
                save_path = str(Path(save_path).with_suffix(".xml"))
            knn.save(save_path)
            print(f"Saved KNN to {save_path}")
    else:
        raise SystemExit("Provide either --train-manifest (to train) or --load-model (to load).")

    print("Evaluating on test set...")
    test_records = read_jsonl(args.test_manifest)
    if args.blue_only:
        test_records = [r for r in test_records if r.get("plate_type") == "blue"]
    if args.max_test_samples:
        test_records = test_records[:args.max_test_samples]

    predictions = []
    targets = []
    t0 = time.time()

    for rec in test_records:
        img_path = rec.get("image_path")
        plate_number = rec.get("plate_number", "")
        if not img_path or not Path(img_path).is_file():
            predictions.append("")
            targets.append(plate_number)
            continue
        img = load_plate(img_path, args.plate_width, args.plate_height)
        if img is None:
            predictions.append("")
            targets.append(plate_number)
            continue
        n_chars = len(plate_number) if plate_number else None
        pred = predict_plate(knn, img, n_chars, args.plate_width, args.plate_height, args.k)
        predictions.append(pred)
        targets.append(plate_number)

    elapsed = time.time() - t0
    metrics = recognition_metrics(predictions, targets)
    metrics.update({
        "samples": len(test_records),
        "elapsed_ms_per_image": elapsed * 1000 / max(len(test_records), 1),
        "fps": len(test_records) / max(elapsed, 1e-6),
        "k": args.k,
    })

    examples_correct = [
        {"target": t, "prediction": p} for p, t in zip(predictions, targets) if p == t
    ][:20]
    examples_wrong = [
        {"target": t, "prediction": p} for p, t in zip(predictions, targets) if p != t
    ][:20]

    result = {
        "model": "KNN_HOG_baseline",
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "metrics": metrics,
        "success_examples": examples_correct,
        "failure_examples": examples_wrong,
        "note": (
            "KNN baseline: HOG features from projection-segmented characters. "
            "Character segmentation quality is the main bottleneck."
        ),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
