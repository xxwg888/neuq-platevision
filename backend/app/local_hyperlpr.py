from __future__ import annotations

import importlib.util
import os
import time
import uuid
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .schemas import CharacterResult, RecognitionImages, RecognitionResult
from .vision import build_binary, crop_with_padding, decode_image, save_image, segment_characters


PLATE_TYPE_NAMES = {
    -1: "unknown",
    0: "blue",
    1: "yellow_single",
    2: "white",
    3: "green",
    4: "black_hk_macao",
    5: "hk_single",
    6: "hk_double",
    7: "macao_single",
    8: "macao_double",
    9: "yellow_double",
}


def is_hyperlpr_available() -> bool:
    if os.getenv("PLATEVISION_DISABLE_HYPERLPR3") == "1":
        return False
    return importlib.util.find_spec("hyperlpr3") is not None


@lru_cache(maxsize=1)
def get_hyperlpr_catcher() -> Any:
    if os.getenv("PLATEVISION_DISABLE_HYPERLPR3") == "1":
        raise RuntimeError("HyperLPR3 is disabled by PLATEVISION_DISABLE_HYPERLPR3=1.")

    import hyperlpr3 as lpr3

    return lpr3.LicensePlateCatcher()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_box(raw_box: Any, width: int, height: int) -> list[int] | None:
    values = np.asarray(raw_box, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return None

    x1, y1, x2, y2 = values[:4]
    left = int(max(min(x1, x2), 0))
    top = int(max(min(y1, y2), 0))
    right = int(min(max(x1, x2), width - 1))
    bottom = int(min(max(y1, y2), height - 1))
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    return [left, top, box_width, box_height]


def _normalize_hyperlpr_result(item: Any, width: int, height: int) -> dict[str, Any] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 4:
        return None

    plate_text = str(item[0]).strip()
    if not plate_text:
        return None

    confidence = max(0.0, min(_as_float(item[1]), 1.0))
    plate_type_code = _as_int(item[2])
    bbox = _normalize_box(item[3], width, height)
    if bbox is None:
        return None

    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "plate_type": PLATE_TYPE_NAMES.get(plate_type_code, "unknown"),
        "bbox": bbox,
    }


def _build_characters(text: str, confidence: float, segmented_chars: list[CharacterResult]) -> list[CharacterResult]:
    chars: list[CharacterResult] = []
    for index, char in enumerate(text):
        bbox = segmented_chars[index].bbox if index < len(segmented_chars) else None
        chars.append(CharacterResult(text=char, confidence=round(confidence, 3), bbox=bbox))
    return chars


def recognize_with_hyperlpr(
    image_bytes: bytes,
    provider: str = "local_model",
    return_intermediate: bool = True,
) -> RecognitionResult:
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    image = decode_image(image_bytes)
    height, width = image.shape[:2]

    detect_start = time.perf_counter()
    raw_results = get_hyperlpr_catcher()(image)
    detect_ms = (time.perf_counter() - detect_start) * 1000

    candidates = [
        normalized
        for item in raw_results
        if (normalized := _normalize_hyperlpr_result(item, width, height)) is not None
    ]
    if not candidates:
        raise RuntimeError("HyperLPR3 did not detect a valid license plate.")

    best = max(candidates, key=lambda item: item["confidence"])
    bbox = best["bbox"]
    x, y, w, h = bbox

    detected = image.copy()
    cv2.rectangle(detected, (x, y), (x + w, y + h), (33, 210, 186), 3)
    cv2.putText(
        detected,
        best["plate_type"].upper(),
        (x, max(y - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (33, 210, 186),
        2,
    )

    plate_crop = crop_with_padding(image, (x, y, w, h))
    binary = build_binary(plate_crop)
    segmented, segmented_chars = segment_characters(plate_crop, binary, request_id)
    chars = _build_characters(best["plate_text"], best["confidence"], segmented_chars)

    image_links = RecognitionImages()
    if return_intermediate:
        image_links.detected = save_image(detected, request_id, "detected")
        image_links.plate_crop = save_image(plate_crop, request_id, "plate")
        image_links.binary = save_image(binary, request_id, "binary")
        image_links.segmented = save_image(segmented, request_id, "segmented")

    total_ms = (time.perf_counter() - start) * 1000
    return RecognitionResult(
        request_id=request_id,
        provider=provider,
        provider_used="hyperlpr3",
        plate_text=best["plate_text"],
        plate_type=best["plate_type"],
        confidence=round(best["confidence"], 3),
        bbox=bbox,
        chars=chars,
        images=image_links,
        timing_ms={
            "detect": round(detect_ms, 2),
            "recognize": round(max(total_ms - detect_ms, 0.0), 2),
            "total": round(total_ms, 2),
        },
        messages=["已使用 HyperLPR3 预训练车牌识别模型完成本地推理。"],
    )
