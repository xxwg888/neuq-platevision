#!/usr/bin/env python
"""Summarize a JSONL manifest for report tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict:
    def count(key: str):
        return dict(Counter(str(r.get(key, "unknown")) for r in records))

    lengths = Counter(len(r.get("plate_number", "")) for r in records)
    provinces = Counter(r.get("plate_number", "")[:1] for r in records if r.get("plate_number"))
    return {
        "count": len(records),
        "source_counts": count("source"),
        "plate_type_counts": count("plate_type"),
        "mode_counts": count("mode"),
        "length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "top_provinces": dict(provinces.most_common(20)),
    }


def main():
    args = parse_args()
    output = {}
    for path in args.manifests:
        output[path] = summarize(read_jsonl(path))
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

