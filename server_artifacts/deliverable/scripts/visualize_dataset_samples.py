#!/usr/bin/env python
"""Create a contact sheet for real license-plate dataset samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scene-width", type=int, default=520)
    parser.add_argument("--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def resize_keep_aspect(img: Image.Image, width: int) -> Image.Image:
    height = max(1, round(img.height * width / img.width))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def draw_poly(draw: ImageDraw.ImageDraw, points: list[list[float]], scale_x: float, scale_y: float):
    pts = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
    draw.line(pts + [pts[0]], fill=(255, 222, 64), width=5)
    draw.line(pts + [pts[0]], fill=(24, 130, 255), width=2)


def choose_records(records: list[dict], max_items: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.get("plate_type", "unknown"), []).append(record)
    for values in grouped.values():
        rng.shuffle(values)

    order = ["blue", "yellow", "white", "green", "black", "unknown"]
    chosen = []
    while len(chosen) < max_items:
        progressed = False
        for key in order:
            values = grouped.get(key, [])
            if values and len(chosen) < max_items:
                chosen.append(values.pop())
                progressed = True
        if not progressed:
            break
    return chosen


def make_tile(record: dict, scene_width: int, fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont]) -> Image.Image:
    title_font, meta_font = fonts
    scene = Image.open(record["scene_path"]).convert("RGB")
    crop = Image.open(record["image_path"]).convert("RGB")
    scene_small = resize_keep_aspect(scene, scene_width)

    draw = ImageDraw.Draw(scene_small)
    scale_x = scene_small.width / scene.width
    scale_y = scene_small.height / scene.height
    draw_poly(draw, record["corners"], scale_x, scale_y)

    crop_width = 230
    crop_small = resize_keep_aspect(crop, crop_width)
    crop_panel_w = crop_width + 24
    text_h = 74
    tile_w = scene_small.width + crop_panel_w
    tile_h = max(scene_small.height, crop_small.height + text_h + 24)
    tile = Image.new("RGB", (tile_w, tile_h), (247, 249, 251))
    tile.paste(scene_small, (0, 0))

    x0 = scene_small.width + 12
    tile.paste(crop_small, (x0, 14))
    tile_draw = ImageDraw.Draw(tile)
    title = record["plate_number"]
    meta = f'{record.get("source", "")} | {record.get("plate_type", "")} | {record.get("mode", "")}'
    tile_draw.text((x0, crop_small.height + 24), title, font=title_font, fill=(20, 26, 34))
    tile_draw.text((x0, crop_small.height + 56), meta, font=meta_font, fill=(81, 93, 107))
    return tile


def main():
    args = parse_args()
    records = read_jsonl(Path(args.manifest))
    chosen = choose_records(records, args.max_items, args.seed)
    fonts = (load_font(args.font, 24), load_font(args.font, 16))
    tiles = [make_tile(record, args.scene_width, fonts) for record in chosen]
    if not tiles:
        raise SystemExit("No records selected.")

    margin = 18
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin), (232, 237, 242))
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        canvas.paste(tile, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)
    print(f"saved {output.resolve()} with {len(tiles)} samples")


if __name__ == "__main__":
    main()
