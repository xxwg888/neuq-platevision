from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .providers import collect_dataset_files, get_providers, parse_ground_truth, recognize_by_provider
from .schemas import (
    BatchEvaluationResult,
    BatchItemResult,
    BatchMetrics,
    RecognitionResult,
    RemoteSettingsPayload,
)
from .settings import EVALUATION_DIR, OUTPUT_DIR, SAMPLES_DIR, RemoteSettings, load_remote_settings, save_remote_settings
from .vision import SUPPORTED_EXTENSIONS


app = FastAPI(
    title="License Plate Recognition Lab API",
    description="OpenCV baseline, local model stub, and remote inference adapter for course design.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_file(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/bmp/webp 图片。")


async def read_upload(file: UploadFile) -> bytes:
    validate_image_file(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单张图片不能超过 12MB。")
    return image_bytes


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def compute_metrics(items: list[BatchItemResult]) -> BatchMetrics:
    labeled = [item for item in items if item.ground_truth]
    if not labeled:
        return BatchMetrics(total=len(items), labeled=0)

    exact_correct = sum(1 for item in labeled if item.correct)
    correct_chars = 0
    pred_chars = 0
    truth_chars = 0
    for item in labeled:
        truth = item.ground_truth or ""
        pred = item.prediction
        distance = edit_distance(truth, pred)
        correct_chars += max(max(len(truth), len(pred)) - distance, 0)
        pred_chars += len(pred)
        truth_chars += len(truth)

    precision = correct_chars / pred_chars if pred_chars else 0.0
    recall = correct_chars / truth_chars if truth_chars else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return BatchMetrics(
        total=len(items),
        labeled=len(labeled),
        exact_accuracy=round(exact_correct / len(labeled), 4),
        char_precision=round(precision, 4),
        char_recall=round(recall, 4),
        char_f1=round(f1, 4),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/providers")
def providers() -> dict[str, object]:
    return {
        "providers": [provider.model_dump() for provider in get_providers()],
        "remote_settings": load_remote_settings().model_dump(),
    }


@app.post("/api/settings/remote")
def update_remote_settings(payload: RemoteSettingsPayload) -> dict[str, object]:
    settings = RemoteSettings(
        enabled=payload.enabled,
        endpoint=payload.endpoint.strip(),
        timeout_seconds=payload.timeout_seconds,
    )
    save_remote_settings(settings)
    return {"ok": True, "remote_settings": settings.model_dump()}


@app.post("/api/recognize", response_model=RecognitionResult)
async def recognize(
    file: UploadFile = File(...),
    provider: str = Form(default="local_model"),
    return_intermediate: bool = Form(default=True),
) -> RecognitionResult:
    image_bytes = await read_upload(file)
    return await recognize_by_provider(image_bytes, file.filename or "upload.jpg", provider, return_intermediate)


@app.post("/api/batch/evaluate", response_model=BatchEvaluationResult)
async def batch_evaluate(
    files: list[UploadFile] | None = File(default=None),
    provider: str = Form(default="local_model"),
    ground_truth: str | None = Form(default=None),
    dataset_dir: str | None = Form(default=None),
    return_intermediate: bool = Form(default=True),
) -> BatchEvaluationResult:
    evaluation_id = f"eval_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    labels = parse_ground_truth(ground_truth)
    results: list[BatchItemResult] = []
    errors: list[BatchItemResult] = []

    work_items: list[tuple[str, bytes]] = []
    if files:
        for file in files[:100]:
            image_bytes = await read_upload(file)
            work_items.append((file.filename or f"upload_{len(work_items)}.jpg", image_bytes))

    try:
        for path in collect_dataset_files(dataset_dir, SAMPLES_DIR):
            work_items.append((path.name, path.read_bytes()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not work_items:
        raise HTTPException(status_code=400, detail="请上传图片或指定 data/samples 下的数据集目录。")

    for file_name, image_bytes in work_items[:100]:
        try:
            result = await recognize_by_provider(image_bytes, file_name, provider, return_intermediate)
            truth = labels.get(file_name) or labels.get(Path(file_name).stem)
            correct = (truth == result.plate_text) if truth is not None else None
            item = BatchItemResult(
                file_name=file_name,
                ground_truth=truth,
                prediction=result.plate_text,
                correct=correct,
                confidence=result.confidence,
                plate_type=result.plate_type,
                output_image=result.images.detected,
                message="; ".join(result.messages) if result.messages else None,
            )
            results.append(item)
            if correct is False:
                errors.append(item)
        except Exception as exc:
            error = BatchItemResult(
                file_name=file_name,
                prediction="",
                confidence=0.0,
                plate_type="unknown",
                message=str(exc),
            )
            results.append(error)
            errors.append(error)

    metrics = compute_metrics(results)
    report_path = EVALUATION_DIR / f"{evaluation_id}.json"
    payload = {
        "evaluation_id": evaluation_id,
        "provider": provider,
        "metrics": metrics.model_dump(),
        "results": [item.model_dump() for item in results],
        "errors": [item.model_dump() for item in errors],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return BatchEvaluationResult(
        evaluation_id=evaluation_id,
        provider=provider,
        metrics=metrics,
        results=results,
        errors=errors,
        report_file=str(report_path),
    )


@app.get("/api/outputs/{file_name}")
def get_output(file_name: str) -> FileResponse:
    path = (OUTPUT_DIR / file_name).resolve()
    if not path.is_relative_to(OUTPUT_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="输出文件不存在。")
    return FileResponse(path)
