#!/usr/bin/env python
"""Build mixed train/val/test manifests from prepared datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed/mixed_plate_ocr/manifests")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_split(paths: list[str], seed: int):
    records: list[dict] = []
    for path in paths:
        records.extend(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {"train": args.train, "val": args.val, "test": args.test}
    summary = {}
    for split, paths in split_paths.items():
        records = load_split(paths, args.seed)
        write_jsonl(output_dir / f"{split}.jsonl", records)
        type_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for record in records:
            type_counts[record.get("plate_type", "unknown")] = type_counts.get(record.get("plate_type", "unknown"), 0) + 1
            source_counts[record.get("source", "unknown")] = source_counts.get(record.get("source", "unknown"), 0) + 1
        summary[split] = {
            "count": len(records),
            "plate_type_counts": type_counts,
            "source_counts": source_counts,
            "inputs": paths,
        }
    (output_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

