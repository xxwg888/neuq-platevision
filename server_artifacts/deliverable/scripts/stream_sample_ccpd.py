#!/usr/bin/env python
"""Stream-sample CCPD2019.tar.xz (forward-only, no full extraction) into a folder.

tar.xz is not randomly seekable, so we open it in streaming mode 'r|xz' and copy a
capped number of .jpg per CCPD split into an output directory. CCPD filenames encode
all labels, so the resulting folder is directly consumable by prepare_ccpd_subset.py.

Usage:
    python scripts/stream_sample_ccpd.py \
        --archive /var/tmp/plate_data_cxj/archives/CCPD2019.tar.xz \
        --out-dir /var/tmp/plate_data_cxj/ccpd2019_sample \
        --per-split 1500 --max-total 12000
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-split", type=int, default=1500, help="max images per ccpd_* split")
    p.add_argument("--max-total", type=int, default=12000)
    p.add_argument("--prefer", default="ccpd_base", help="split to fill first / most")
    return p.parse_args()


def split_of(name: str) -> str:
    parts = name.split("/")
    for part in parts:
        if part.startswith("ccpd_"):
            return part
    return "other"


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_split_counts: dict[str, int] = {}
    total = 0
    seen = 0

    with tarfile.open(args.archive, "r|xz") as tar:
        for member in tar:
            if total >= args.max_total:
                break
            if not member.isfile():
                continue
            name = member.name
            if not name.lower().endswith(".jpg"):
                continue
            seen += 1
            sp = split_of(name)
            cap = args.per_split
            if sp == args.prefer:
                cap = args.per_split * 3  # keep more of the clean base split
            if per_split_counts.get(sp, 0) >= cap:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            # flatten name: <split>__<basename>
            base = Path(name).name
            dest = out / f"{sp}__{base}"
            dest.write_bytes(data)
            per_split_counts[sp] = per_split_counts.get(sp, 0) + 1
            total += 1
            if total % 1000 == 0:
                print(f"saved={total} seen={seen} splits={per_split_counts}", flush=True)

    print(f"DONE saved={total} seen={seen}")
    print(f"per_split={per_split_counts}")


if __name__ == "__main__":
    main()
