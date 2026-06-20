from __future__ import annotations

import importlib.util
import os
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .schemas import CharacterResult, RecognitionImages, RecognitionResult
from .settings import PROJECT_ROOT
from .vision import build_binary, crop_with_padding, decode_image, save_image, segment_characters


ARTIFACT_ROOT = PROJECT_ROOT / "server_artifacts" / "deliverable"
POSE_ONNX = ARTIFACT_ROOT / "models" / "yolo_pose" / "best.onnx"
OCR_ONNX = ARTIFACT_ROOT / "models" / "ocr" / "ocr_best.onnx"
CHARSET_FILE = ARTIFACT_ROOT / "charset.txt"
CLASS_NAMES = ["blue", "green", "yellow_single", "yellow_double", "white", "black"]


@dataclass
class PlateDetection:
    box: list[float]
    confidence: float
    plate_type: str
    corners: list[list[float]]


@dataclass
class OnnxRuntimeBundle:
    pose_session: object
    ocr_session: object
    charset: list[str]


def is_trained_onnx_available() -> bool:
    if os.getenv("PLATEVISION_DISABLE_TRAINED_ONNX") == "1":
        return False
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and POSE_ONNX.exists()
        and OCR_ONNX.exists()
        and CHARSET_FILE.exists()
    )


@lru_cache(maxsize=1)
def get_runtime_bundle() -> OnnxRuntimeBundle:
    if not is_trained_onnx_available():
        raise RuntimeError("self-trained ONNX artifacts are unavailable.")

    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    pose_session = ort.InferenceSession(str(POSE_ONNX), providers=providers)
    ocr_session = ort.InferenceSession(str(OCR_ONNX), providers=providers)
    charset = CHARSET_FILE.read_text(encoding="utf-8").splitlines()
    return OnnxRuntimeBundle(pose_session=pose_session, ocr_session=ocr_session, charset=charset)


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_plate(image: np.ndarray, corners: list[list[float]], width: int = 320, height: int = 96) -> np.ndarray:
    source = _order_points(np.asarray(corners, dtype=np.float32))
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (width, height))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _ctc_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, list[float]]:
    probabilities = _softmax(logits[0])
    sequence = probabilities.argmax(axis=-1).tolist()
    text: list[str] = []
    confidences: list[float] = []
    previous = -1
    for timestep, index in enumerate(sequence):
        if index != previous and index != 0:
            text.append(charset[index])
            confidences.append(float(probabilities[timestep, index]))
        previous = index
    return "".join(text), confidences


def _detect_pose_onnx(session: object, image: np.ndarray, image_size: int = 640) -> list[PlateDetection]:
    image_height, image_width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size)).astype(np.float32) / 255.0
    model_input = np.transpose(resized, (2, 0, 1))[None].astype(np.float32)
    output = session.run(None, {session.get_inputs()[0].name: model_input})[0]
    predictions = np.squeeze(output, axis=0)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose(1, 0)

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4 : 4 + len(CLASS_NAMES)]
    keypoints = predictions[:, 4 + len(CLASS_NAMES) :]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    keep = scores >= 0.25
    boxes = boxes[keep]
    keypoints = keypoints[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]
    if len(boxes) == 0:
        return []

    scale_x = image_width / image_size
    scale_y = image_height / image_size
    nms_boxes: list[list[float]] = []
    metadata: list[tuple[float, int, list[list[float]]]] = []
    for (center_x, center_y, box_width, box_height), keypoint, class_id, score in zip(
        boxes,
        keypoints,
        class_ids,
        scores,
    ):
        x = float((center_x - box_width / 2) * scale_x)
        y = float((center_y - box_height / 2) * scale_y)
        w = float(box_width * scale_x)
        h = float(box_height * scale_y)
        corners = [
            [float(keypoint[3 * index] * scale_x), float(keypoint[3 * index + 1] * scale_y)]
            for index in range(4)
        ]
        nms_boxes.append([x, y, w, h])
        metadata.append((float(score), int(class_id), corners))

    indices = cv2.dnn.NMSBoxes(nms_boxes, [item[0] for item in metadata], 0.25, 0.45)
    detections: list[PlateDetection] = []
    for index in np.array(indices).flatten() if len(indices) else []:
        score, class_id, corners = metadata[index]
        detections.append(
            PlateDetection(
                box=nms_boxes[index],
                confidence=score,
                plate_type=CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
                corners=corners,
            )
        )
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def _recognize_ocr_onnx(session: object, crop: np.ndarray, charset: list[str]) -> tuple[str, list[float]]:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 48)).astype(np.float32) / 255.0
    normalized = (resized - 0.5) / 0.5
    model_input = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: model_input})[0]
    return _ctc_decode(logits, charset)


def _draw_detection(image: np.ndarray, detection: PlateDetection) -> np.ndarray:
    canvas = image.copy()
    x, y, w, h = [int(round(value)) for value in detection.box]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (33, 210, 186), 3)
    points = np.asarray(detection.corners, dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 210, 255), thickness=2)
    for point_x, point_y in points:
        cv2.circle(canvas, (int(point_x), int(point_y)), 4, (0, 0, 255), -1)
    cv2.putText(
        canvas,
        detection.plate_type.upper(),
        (x, max(y - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (33, 210, 186),
        2,
    )
    return canvas


def _build_characters(
    text: str,
    confidences: list[float],
    segmented_chars: list[CharacterResult],
) -> list[CharacterResult]:
    characters: list[CharacterResult] = []
    fallback = float(np.mean(confidences)) if confidences else 0.0
    for index, char in enumerate(text):
        confidence = confidences[index] if index < len(confidences) else fallback
        bbox = segmented_chars[index].bbox if index < len(segmented_chars) else None
        characters.append(CharacterResult(text=char, confidence=round(float(confidence), 3), bbox=bbox))
    return characters


def recognize_with_trained_onnx(
    image_bytes: bytes,
    provider: str = "local_model",
    return_intermediate: bool = True,
) -> RecognitionResult:
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    image = decode_image(image_bytes)
    runtime = get_runtime_bundle()

    detect_start = time.perf_counter()
    detections = _detect_pose_onnx(runtime.pose_session, image)
    detect_ms = (time.perf_counter() - detect_start) * 1000
    if not detections:
        raise RuntimeError("self-trained ONNX detector did not find a license plate.")

    detection = detections[0]
    plate_crop = _warp_plate(image, detection.corners)
    plate_text, char_confidences = _recognize_ocr_onnx(runtime.ocr_session, plate_crop, runtime.charset)
    if not plate_text:
        raise RuntimeError("self-trained ONNX OCR returned an empty plate string.")

    binary = build_binary(plate_crop)
    segmented, segmented_chars = segment_characters(plate_crop, binary, request_id)
    chars = _build_characters(plate_text, char_confidences, segmented_chars)
    ocr_confidence = float(np.mean(char_confidences)) if char_confidences else detection.confidence
    confidence = round(float(detection.confidence * ocr_confidence), 3)

    image_links = RecognitionImages()
    if return_intermediate:
        detected = _draw_detection(image, detection)
        image_links.detected = save_image(detected, request_id, "detected")
        image_links.plate_crop = save_image(plate_crop, request_id, "plate")
        image_links.binary = save_image(binary, request_id, "binary")
        image_links.segmented = save_image(segmented, request_id, "segmented")

    x, y, w, h = detection.box
    total_ms = (time.perf_counter() - start) * 1000
    return RecognitionResult(
        request_id=request_id,
        provider=provider,
        provider_used="trained_onnx",
        plate_text=plate_text,
        plate_type=detection.plate_type,
        confidence=confidence,
        bbox=[int(round(x)), int(round(y)), int(round(w)), int(round(h))],
        chars=chars,
        images=image_links,
        timing_ms={
            "detect": round(detect_ms, 2),
            "recognize": round(max(total_ms - detect_ms, 0.0), 2),
            "total": round(total_ms, 2),
        },
        messages=[
            "已使用服务器自训练 YOLOv8n-pose + CRNN-CTC ONNX 模型完成本地 CPU 推理。",
            "二值化和字符框仅用于过程展示；CRNN-CTC 对透视校正后的整块车牌进行序列识别。",
        ],
    )
