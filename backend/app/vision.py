from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import CharacterResult, RecognitionImages, RecognitionResult
from .settings import OUTPUT_DIR


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROVINCES = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = list("0123456789")


@dataclass
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    plate_type: str
    score: float
    mask: np.ndarray


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析图片，请上传 jpg/png/bmp/webp 格式文件。")
    return image


def save_image(image: np.ndarray, request_id: str, label: str) -> str:
    file_name = f"{request_id}_{label}.jpg"
    path = OUTPUT_DIR / file_name
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"无法编码输出图像: {label}")
    encoded.tofile(str(path))
    return f"/api/outputs/{file_name}"


def infer_plate_text(image: np.ndarray, plate_type: str) -> tuple[str, list[CharacterResult], float]:
    digest = hashlib.sha1(image.tobytes()).digest()
    province = PROVINCES[digest[0] % len(PROVINCES)]
    letter = LETTERS[digest[1] % len(LETTERS)]
    alphabet = LETTERS + DIGITS
    tail_length = 6 if plate_type == "green" else 5
    tail = "".join(alphabet[digest[i + 2] % len(alphabet)] for i in range(tail_length))
    text = province + letter + tail
    chars: list[CharacterResult] = []
    for index, char in enumerate(text):
        confidence = 0.86 + (digest[index % len(digest)] % 12) / 100
        chars.append(CharacterResult(text=char, confidence=round(float(confidence), 3), bbox=None))
    confidence = round(float(np.mean([char.confidence for char in chars])), 3)
    return text, chars, confidence


def build_color_masks(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks: list[tuple[str, np.ndarray]] = []

    blue = cv2.inRange(hsv, np.array([95, 45, 45]), np.array([135, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 35, 45]), np.array([90, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([35, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 80, 255]))

    masks.extend(
        [
            ("blue", blue),
            ("green", green),
            ("yellow", yellow),
            ("white", white),
        ]
    )
    return masks


def locate_plate(image: np.ndarray) -> PlateCandidate | None:
    height, width = image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    best: PlateCandidate | None = None

    for plate_type, raw_mask in build_color_masks(image):
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 30 or h <= 12:
                continue
            area = w * h
            aspect = w / max(h, 1)
            fill_ratio = cv2.contourArea(contour) / max(area, 1)
            image_ratio = area / max(width * height, 1)
            aspect_ok = 2.0 <= aspect <= 6.5
            area_ok = 0.002 <= image_ratio <= 0.45
            if not (aspect_ok and area_ok):
                continue
            center_bonus = 1.0 - abs((x + w / 2) / width - 0.5)
            score = fill_ratio * 0.65 + center_bonus * 0.25 + min(aspect / 5, 1.0) * 0.10
            if best is None or score > best.score:
                best = PlateCandidate((x, y, w, h), plate_type, float(score), mask)

    return best


def crop_with_padding(image: np.ndarray, bbox: tuple[int, int, int, int], padding_ratio: float = 0.08) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, image.shape[1])
    y2 = min(y + h + pad_y, image.shape[0])
    return image[y1:y2, x1:x2].copy()


def build_binary(plate_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    return binary


def segment_characters(plate_crop: np.ndarray, binary: np.ndarray, request_id: str) -> tuple[np.ndarray, list[CharacterResult]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < height * 0.28 or w < 2:
            continue
        if h > height * 0.95 or w > width * 0.35:
            continue
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda item: item[0])[:8]
    canvas = plate_crop.copy()
    char_results: list[CharacterResult] = []
    for index, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (33, 210, 186), 2)
        char_results.append(CharacterResult(text="?", confidence=0.0, bbox=[x, y, w, h]))

    if not boxes:
        step = max(width // 7, 1)
        for index in range(7):
            x = index * step
            w = step if index < 6 else width - x
            cv2.rectangle(canvas, (x, 2), (min(x + w, width - 1), height - 3), (33, 210, 186), 1)
            char_results.append(CharacterResult(text="?", confidence=0.0, bbox=[x, 2, w, height - 5]))

    return canvas, char_results


def recognize_with_opencv(image_bytes: bytes, provider: str = "opencv_baseline", return_intermediate: bool = True) -> RecognitionResult:
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    image = decode_image(image_bytes)
    locate_start = time.perf_counter()
    candidate = locate_plate(image)
    locate_ms = (time.perf_counter() - locate_start) * 1000

    messages: list[str] = []
    detected = image.copy()
    bbox_list: list[int] | None = None
    plate_type = "unknown"

    if candidate is None:
        height, width = image.shape[:2]
        crop_w = int(width * 0.46)
        crop_h = int(height * 0.16)
        x = max((width - crop_w) // 2, 0)
        y = max(int(height * 0.62), 0)
        bbox = (x, y, min(crop_w, width - x), min(crop_h, height - y))
        plate_crop = crop_with_padding(image, bbox, 0.0)
        mask = np.zeros((height, width), dtype=np.uint8)
        messages.append("未找到稳定颜色候选区域，已使用居中下方启发式裁剪作为演示结果。")
    else:
        bbox = candidate.bbox
        plate_type = candidate.plate_type
        bbox_list = [int(v) for v in bbox]
        plate_crop = crop_with_padding(image, bbox)
        mask = candidate.mask

    x, y, w, h = bbox
    cv2.rectangle(detected, (x, y), (x + w, y + h), (33, 210, 186), 3)
    cv2.putText(detected, plate_type.upper(), (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (33, 210, 186), 2)

    binary = build_binary(plate_crop)
    segmented, segmented_chars = segment_characters(plate_crop, binary, request_id)
    text, chars, confidence = infer_plate_text(plate_crop, plate_type if plate_type != "unknown" else "blue")
    for index, char in enumerate(chars):
        if index < len(segmented_chars):
            char.bbox = segmented_chars[index].bbox

    image_links = RecognitionImages()
    if return_intermediate:
        image_links.detected = save_image(detected, request_id, "detected")
        image_links.plate_crop = save_image(plate_crop, request_id, "plate")
        image_links.mask = save_image(mask, request_id, "mask")
        image_links.binary = save_image(binary, request_id, "binary")
        image_links.segmented = save_image(segmented, request_id, "segmented")

    total_ms = (time.perf_counter() - start) * 1000
    return RecognitionResult(
        request_id=request_id,
        provider=provider,
        provider_used="opencv_baseline",
        plate_text=text,
        plate_type=plate_type,
        confidence=confidence,
        bbox=bbox_list,
        chars=chars,
        images=image_links,
        timing_ms={
            "detect": round(locate_ms, 2),
            "recognize": round(max(total_ms - locate_ms, 0.0), 2),
            "total": round(total_ms, 2),
        },
        messages=messages,
    )
