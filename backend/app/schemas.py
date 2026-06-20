from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    mode: str


class CharacterResult(BaseModel):
    text: str
    confidence: float
    bbox: list[int] | None = None


class RecognitionImages(BaseModel):
    detected: str | None = None
    plate_crop: str | None = None
    mask: str | None = None
    binary: str | None = None
    segmented: str | None = None


class RecognitionResult(BaseModel):
    request_id: str
    provider: str
    provider_used: str
    plate_text: str
    plate_type: str
    confidence: float
    bbox: list[int] | None
    chars: list[CharacterResult]
    images: RecognitionImages
    timing_ms: dict[str, float]
    messages: list[str] = Field(default_factory=list)


class RemoteSettingsPayload(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    timeout_seconds: float = Field(default=12.0, ge=1.0, le=120.0)


class BatchItemResult(BaseModel):
    file_name: str
    ground_truth: str | None = None
    prediction: str
    correct: bool | None = None
    confidence: float
    plate_type: str
    output_image: str | None = None
    message: str | None = None


class BatchMetrics(BaseModel):
    total: int
    labeled: int
    exact_accuracy: float | None = None
    char_precision: float | None = None
    char_recall: float | None = None
    char_f1: float | None = None


class BatchEvaluationResult(BaseModel):
    evaluation_id: str
    provider: str
    metrics: BatchMetrics
    results: list[BatchItemResult]
    errors: list[BatchItemResult]
    report_file: str | None = None

