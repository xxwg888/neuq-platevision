from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .schemas import CharacterResult, RecognitionImages, RecognitionResult
from .settings import PROJECT_ROOT
from .trained_onnx import (
    ARTIFACT_ROOT,
    PlateDetection,
    _detect_pose_onnx,
    _draw_detection,
    _warp_plate,
    get_runtime_bundle,
    is_trained_onnx_available,
)
from .vision import build_color_masks, crop_with_padding, decode_image, locate_plate, save_image


KNN_MODEL = ARTIFACT_ROOT / "models" / "knn" / "knn_baseline.xml"
TEMPLATE_MODEL = ARTIFACT_ROOT / "models" / "template" / "templates.npz"
CACHE_ROOT = Path(os.getenv("PLATEVISION_CACHE_DIR", r"C:\platevision_cache"))

PLATE_W = 160
PLATE_H = 48
CHAR_W = 24
CHAR_H = 40
HOG = cv2.HOGDescriptor((CHAR_W, CHAR_H), (16, 16), (8, 8), (8, 8), 9)

PROVINCES = list("京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳")
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("I", "O")]
DIGITS = list("0123456789")
SPECIALS = list("警学挂领使临")
ALL_CHARS = PROVINCES + LETTERS + DIGITS + SPECIALS
IDX_TO_CHAR = {index: char for index, char in enumerate(ALL_CHARS)}


@dataclass
class PreparedPlate:
    crop: np.ndarray
    normalized: np.ndarray
    detected: np.ndarray
    plate_type: str
    bbox: list[int] | None
    detection_confidence: float
    detect_ms: float


@dataclass
class BinaryPlate:
    image: np.ndarray
    offset_x: int
    offset_y: int


@dataclass
class Segment:
    patch: np.ndarray
    bbox: list[int]


@dataclass
class TemplateBundle:
    classes: list[str]
    templates: np.ndarray


def is_traditional_knn_available() -> bool:
    return KNN_MODEL.exists()


def is_traditional_ncc_available() -> bool:
    return TEMPLATE_MODEL.exists()


def _ascii_cache_path(source: Path) -> Path:
    source = source.resolve()
    if str(source).isascii():
        return source

    target = CACHE_ROOT / "models" / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target


@lru_cache(maxsize=1)
def _load_knn() -> cv2.ml_KNearest:
    if not KNN_MODEL.exists():
        raise RuntimeError("HOG+KNN model is missing: models/knn/knn_baseline.xml")

    model_path = _ascii_cache_path(KNN_MODEL)
    knn = cv2.ml.KNearest_load(str(model_path))
    if knn.empty():
        raise RuntimeError(f"OpenCV failed to load KNN model: {model_path}")
    knn.setDefaultK(5)
    return knn


@lru_cache(maxsize=1)
def _load_templates() -> TemplateBundle:
    if not TEMPLATE_MODEL.exists():
        raise RuntimeError("NCC template model is missing: models/template/templates.npz")

    payload = np.load(TEMPLATE_MODEL, allow_pickle=True)
    classes = [str(item) for item in payload["classes"].tolist()]
    templates = payload["templates"].astype(np.float32)
    return TemplateBundle(classes=classes, templates=templates)


def _smooth1d(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _binarize_plate(plate: np.ndarray) -> BinaryPlate:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_y = int(round(height * 0.10))
    margin_x = int(round(width * 0.02))
    if height - 2 * margin_y > 8 and width - 2 * margin_x > 8:
        gray = gray[margin_y : height - margin_y, margin_x : width - margin_x]
    else:
        margin_x = 0
        margin_y = 0

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return BinaryPlate(binary, margin_x, margin_y)


def _segment_to_n(binary: BinaryPlate, n_chars: int) -> list[Segment]:
    bw = binary.image
    height, width = bw.shape
    projection = _smooth1d(bw.sum(axis=0), 5)
    threshold = projection.max() * 0.10 if projection.size else 0
    runs: list[list[int]] = []
    start = 0
    in_char = False

    for x in range(width):
        if not in_char and projection[x] > threshold:
            start = x
            in_char = True
        elif in_char and projection[x] <= threshold:
            if x - start >= 2:
                runs.append([start, x])
            in_char = False
    if in_char and width - start >= 2:
        runs.append([start, width])

    if not runs:
        step = max(width // max(n_chars, 1), 1)
        runs = [[index * step, (index + 1) * step if index < n_chars - 1 else width] for index in range(n_chars)]

    while len(runs) > n_chars:
        gaps = [(runs[index + 1][0] - runs[index][1], index) for index in range(len(runs) - 1)]
        _, merge_at = min(gaps)
        runs[merge_at] = [runs[merge_at][0], runs[merge_at + 1][1]]
        del runs[merge_at + 1]

    while len(runs) < n_chars:
        widths = [(runs[index][1] - runs[index][0], index) for index in range(len(runs))]
        _, split_at = max(widths)
        start_x, end_x = runs[split_at]
        if end_x - start_x < 2:
            break
        mid = (start_x + end_x) // 2
        runs[split_at] = [start_x, mid]
        runs.insert(split_at + 1, [mid, end_x])

    segments: list[Segment] = []
    for start_x, end_x in runs[:n_chars]:
        if end_x <= start_x:
            continue
        x = int(start_x + binary.offset_x)
        y = int(binary.offset_y)
        segments.append(Segment(bw[:, start_x:end_x], [x, y, int(end_x - start_x), int(height)]))
    return segments


def _hog_features(patch: np.ndarray) -> np.ndarray:
    resized = cv2.resize(patch, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
    return HOG.compute(resized).flatten()


def _resized_glyph(patch: np.ndarray) -> np.ndarray:
    resized = cv2.resize(patch, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _ncc_vector(glyph: np.ndarray) -> np.ndarray:
    vector = glyph.flatten()
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else vector


def _expected_char_count(plate_type: str) -> int:
    return 8 if plate_type == "green" else 7


def _estimate_plate_type(image: np.ndarray) -> tuple[str, float]:
    counts = {name: int(mask.sum() // 255) for name, mask in build_color_masks(image)}
    if not counts:
        return "blue", 0.0
    best_type, best_count = max(counts.items(), key=lambda item: item[1])
    area = max(image.shape[0] * image.shape[1], 1)
    if best_type == "yellow":
        best_type = "yellow_single"
    return best_type, best_count / area


def _prepare_plate(image: np.ndarray) -> PreparedPlate:
    start = time.perf_counter()
    height, width = image.shape[:2]
    aspect = width / max(height, 1)
    direct_type, color_coverage = _estimate_plate_type(image)
    if 2.0 <= aspect <= 6.5 and color_coverage >= 0.18:
        detected = image.copy()
        cv2.rectangle(detected, (1, 1), (width - 2, height - 2), (33, 210, 186), 2)
        cv2.putText(detected, direct_type.upper(), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (33, 210, 186), 2)
        return PreparedPlate(
            crop=image.copy(),
            normalized=cv2.resize(image, (PLATE_W, PLATE_H), interpolation=cv2.INTER_AREA),
            detected=detected,
            plate_type=direct_type,
            bbox=[0, 0, int(width), int(height)],
            detection_confidence=0.92,
            detect_ms=(time.perf_counter() - start) * 1000,
        )

    if is_trained_onnx_available():
        try:
            runtime = get_runtime_bundle()
            detections = _detect_pose_onnx(runtime.pose_session, image)
            if detections:
                detection: PlateDetection = detections[0]
                crop = _warp_plate(image, detection.corners)
                detected = _draw_detection(image, detection)
                x, y, w, h = detection.box
                return PreparedPlate(
                    crop=crop,
                    normalized=cv2.resize(crop, (PLATE_W, PLATE_H), interpolation=cv2.INTER_AREA),
                    detected=detected,
                    plate_type=detection.plate_type,
                    bbox=[int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                    detection_confidence=detection.confidence,
                    detect_ms=(time.perf_counter() - start) * 1000,
                )
        except Exception:
            pass

    candidate = locate_plate(image)
    detected = image.copy()
    if candidate is None:
        height, width = image.shape[:2]
        crop_w = int(width * 0.46)
        crop_h = int(height * 0.16)
        x = max((width - crop_w) // 2, 0)
        y = max(int(height * 0.62), 0)
        bbox = (x, y, min(crop_w, width - x), min(crop_h, height - y))
        crop = crop_with_padding(image, bbox, 0.0)
        plate_type = "blue"
        score = 0.35
    else:
        bbox = candidate.bbox
        crop = crop_with_padding(image, bbox)
        plate_type = candidate.plate_type
        score = candidate.score

    x, y, w, h = bbox
    cv2.rectangle(detected, (x, y), (x + w, y + h), (33, 210, 186), 3)
    cv2.putText(detected, plate_type.upper(), (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (33, 210, 186), 2)
    return PreparedPlate(
        crop=crop,
        normalized=cv2.resize(crop, (PLATE_W, PLATE_H), interpolation=cv2.INTER_AREA),
        detected=detected,
        plate_type=plate_type,
        bbox=[int(v) for v in bbox],
        detection_confidence=float(score),
        detect_ms=(time.perf_counter() - start) * 1000,
    )


def _predict_knn(segments: list[Segment]) -> tuple[str, list[float]]:
    knn = _load_knn()
    text: list[str] = []
    confidences: list[float] = []
    for segment in segments:
        feature = np.array([_hog_features(segment.patch)], dtype=np.float32)
        _, results, neighbours, _ = knn.findNearest(feature, k=5)
        index = int(results[0][0])
        label = IDX_TO_CHAR.get(index, "?")
        votes = neighbours.flatten().astype(int).tolist() if neighbours is not None else []
        confidence = votes.count(index) / max(len(votes), 1) if votes else 0.0
        text.append(label)
        confidences.append(float(confidence))
    return "".join(text), confidences


def _predict_ncc(segments: list[Segment]) -> tuple[str, list[float]]:
    bundle = _load_templates()
    text: list[str] = []
    confidences: list[float] = []
    for segment in segments:
        vector = _ncc_vector(_resized_glyph(segment.patch)).astype(np.float32)
        scores = bundle.templates @ vector
        best = int(np.argmax(scores))
        score = float(scores[best])
        text.append(bundle.classes[best])
        confidences.append(max(min((score + 1.0) / 2.0, 1.0), 0.0))
    return "".join(text), confidences


def _draw_segments(plate: np.ndarray, segments: list[Segment], labels: list[str]) -> np.ndarray:
    canvas = plate.copy()
    for index, segment in enumerate(segments):
        x, y, w, h = segment.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (33, 210, 186), 1)
        label = labels[index] if index < len(labels) and labels[index].isascii() else str(index + 1)
        cv2.putText(canvas, label, (x + 1, max(y - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
    return cv2.resize(canvas, (320, 96), interpolation=cv2.INTER_NEAREST)


def _draw_classifier_view(binary: BinaryPlate, segments: list[Segment], labels: list[str]) -> np.ndarray:
    canvas = cv2.cvtColor(binary.image, cv2.COLOR_GRAY2BGR)
    for index, segment in enumerate(segments):
        x = segment.bbox[0] - binary.offset_x
        y = 0
        w = segment.bbox[2]
        h = binary.image.shape[0]
        cv2.rectangle(canvas, (x, y), (x + w, h - 1), (33, 210, 186), 1)
        label = labels[index] if index < len(labels) and labels[index].isascii() else str(index + 1)
        cv2.putText(canvas, label, (x + 1, min(h - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
    return cv2.resize(canvas, (320, 96), interpolation=cv2.INTER_NEAREST)


def recognize_with_traditional_baseline(
    image_bytes: bytes,
    provider: str,
    method: str,
    return_intermediate: bool = True,
) -> RecognitionResult:
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    image = decode_image(image_bytes)
    prepared = _prepare_plate(image)
    binary = _binarize_plate(prepared.normalized)
    n_chars = _expected_char_count(prepared.plate_type)
    segments = _segment_to_n(binary, n_chars)
    if not segments:
        raise RuntimeError("traditional baseline failed to segment plate characters.")

    recognize_start = time.perf_counter()
    if method == "knn":
        plate_text, confidences = _predict_knn(segments)
        provider_used = "hog_knn"
        method_label = "HOG+KNN"
    elif method == "ncc":
        plate_text, confidences = _predict_ncc(segments)
        provider_used = "template_ncc"
        method_label = "Template NCC"
    else:
        raise ValueError(f"unknown traditional method: {method}")
    recognize_ms = (time.perf_counter() - recognize_start) * 1000

    labels = list(plate_text)
    chars = [
        CharacterResult(text=char, confidence=round(confidences[index] if index < len(confidences) else 0.0, 3), bbox=segment.bbox)
        for index, (char, segment) in enumerate(zip(labels, segments))
    ]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    confidence = round(float(confidence * prepared.detection_confidence), 3)

    images = RecognitionImages()
    if return_intermediate:
        segmented = _draw_segments(prepared.normalized, segments, labels)
        classifier = _draw_classifier_view(binary, segments, labels)
        images.detected = save_image(prepared.detected, request_id, "detected")
        images.plate_crop = save_image(prepared.crop, request_id, "plate")
        images.binary = save_image(binary.image, request_id, "binary")
        images.segmented = save_image(segmented, request_id, "segmented")
        images.mask = save_image(classifier, request_id, "classifier")

    total_ms = (time.perf_counter() - start) * 1000
    return RecognitionResult(
        request_id=request_id,
        provider=provider,
        provider_used=provider_used,
        plate_text=plate_text,
        plate_type=prepared.plate_type,
        confidence=confidence,
        bbox=prepared.bbox,
        chars=chars,
        images=images,
        timing_ms={
            "detect": round(prepared.detect_ms, 2),
            "recognize": round(recognize_ms, 2),
            "total": round(total_ms, 2),
        },
        messages=[
            f"已使用传统 {method_label} baseline：先定位并透视校正车牌，再进行二值化、投影字符分割和逐字符分类。",
            "该路线用于课程设计中的传统方法对比；CRNN-CTC 路线不依赖手工字符分割。",
        ],
    )
