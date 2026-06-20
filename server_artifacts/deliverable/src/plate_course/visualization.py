"""Visualization helpers for real-data preprocessing scripts."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_overview(records: list[dict], output_path: Path, max_items: int = 24) -> None:
    items = records[:max_items]
    thumb_w, thumb_h = 220, 80
    cols = 4
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 26)), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font = get_font(16)

    for idx, rec in enumerate(items):
        img = Image.open(rec["image_path"]).convert("RGB")
        img.thumbnail((thumb_w - 12, thumb_h - 8))
        x = (idx % cols) * thumb_w + 6
        y = (idx // cols) * (thumb_h + 26) + 4
        canvas.paste(img, (x, y))
        draw.text((x, y + thumb_h), f'{rec["plate_number"]} | {rec["plate_type"]}', font=font, fill=(30, 35, 45))

    canvas.save(output_path)
