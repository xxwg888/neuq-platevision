from __future__ import annotations

import json
from pathlib import Path

import httpx

from .local_hyperlpr import is_hyperlpr_available, recognize_with_hyperlpr
from .schemas import ProviderInfo, RecognitionResult
from .settings import RemoteSettings, load_remote_settings
from .traditional_baseline import (
    is_traditional_knn_available,
    is_traditional_ncc_available,
    recognize_with_traditional_baseline,
)
from .trained_onnx import is_trained_onnx_available, recognize_with_trained_onnx
from .vision import SUPPORTED_EXTENSIONS, recognize_with_opencv


def get_providers() -> list[ProviderInfo]:
    remote = load_remote_settings()
    trained_available = is_trained_onnx_available()
    hyperlpr_available = is_hyperlpr_available()
    knn_available = is_traditional_knn_available()
    ncc_available = is_traditional_ncc_available()
    return [
        ProviderInfo(
            id="local_model",
            name="Self-trained ONNX Model",
            description="服务器自训练 YOLOv8n-pose + CRNN-CTC ONNX，本地 CPU 推理；主实验方法。",
            available=trained_available,
            mode="local_trained_onnx" if trained_available else "local_stub",
        ),
        ProviderInfo(
            id="pretrained_hyperlpr",
            name="HyperLPR3 Pretrained Model",
            description="HyperLPR3 预训练车牌识别模块，用于和自训练模型、传统 baseline 做效果对比。",
            available=hyperlpr_available,
            mode="local_pretrained" if hyperlpr_available else "local_stub",
        ),
        ProviderInfo(
            id="traditional_knn",
            name="Traditional HOG+KNN Baseline",
            description="传统对比方法：定位/校正后进行二值化、投影字符分割、HOG 特征提取和 KNN 分类。",
            available=knn_available,
            mode="traditional_knn",
        ),
        ProviderInfo(
            id="traditional_ncc",
            name="Traditional NCC Baseline",
            description="传统对比方法：定位/校正后进行二值化、投影字符分割，并用字符模板 NCC 相似度分类。",
            available=ncc_available,
            mode="traditional_ncc",
        ),
        ProviderInfo(
            id="remote_server",
            name="Remote GPU Server",
            description="预留老师服务器推理接口；未配置时回退到 OpenCV baseline。",
            available=remote.enabled and bool(remote.endpoint),
            mode="remote",
        ),
    ]


async def call_remote_provider(
    image_bytes: bytes,
    file_name: str,
    return_intermediate: bool,
    settings: RemoteSettings,
) -> RecognitionResult:
    if not settings.enabled or not settings.endpoint:
        result = recognize_with_opencv(image_bytes, provider="remote_server", return_intermediate=return_intermediate)
        result.provider_used = "opencv_baseline"
        result.messages.append("远程推理尚未启用，已自动回退到本地 OpenCV baseline。")
        return result

    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            response = await client.post(
                settings.endpoint,
                data={"provider": "remote_server", "return_intermediate": str(return_intermediate).lower()},
                files={"file": (file_name, image_bytes, "application/octet-stream")},
            )
            response.raise_for_status()
            result = RecognitionResult.model_validate(response.json())
            result.provider = "remote_server"
            result.provider_used = "remote_server"
            return result
    except (httpx.HTTPError, ValueError) as exc:
        result = recognize_with_opencv(image_bytes, provider="remote_server", return_intermediate=return_intermediate)
        result.provider_used = "opencv_baseline"
        result.messages.append(f"远程推理调用失败，已回退到本地 baseline：{exc}")
        return result


async def recognize_by_provider(
    image_bytes: bytes,
    file_name: str,
    provider: str,
    return_intermediate: bool,
) -> RecognitionResult:
    normalized = provider or "opencv_baseline"
    if normalized == "remote_server":
        return await call_remote_provider(image_bytes, file_name, return_intermediate, load_remote_settings())
    if normalized == "traditional_knn":
        return recognize_with_traditional_baseline(
            image_bytes,
            provider=normalized,
            method="knn",
            return_intermediate=return_intermediate,
        )
    if normalized == "traditional_ncc":
        return recognize_with_traditional_baseline(
            image_bytes,
            provider=normalized,
            method="ncc",
            return_intermediate=return_intermediate,
        )
    if normalized == "pretrained_hyperlpr":
        return recognize_with_hyperlpr(image_bytes, provider=normalized, return_intermediate=return_intermediate)
    if normalized == "local_model":
        onnx_error: Exception | None = None
        try:
            return recognize_with_trained_onnx(image_bytes, provider=normalized, return_intermediate=return_intermediate)
        except Exception as exc:
            onnx_error = exc

        try:
            result = recognize_with_hyperlpr(image_bytes, provider=normalized, return_intermediate=return_intermediate)
            result.messages.insert(0, f"自训练 ONNX 模型不可用或未检出，已回退到 HyperLPR3：{onnx_error}")
            return result
        except Exception as hyperlpr_exc:
            result = recognize_with_opencv(image_bytes, provider=normalized, return_intermediate=return_intermediate)
            result.provider_used = "opencv_baseline"
            result.messages.append(
                "自训练 ONNX 与 HyperLPR3 均不可用或未检出车牌，已回退到 OpenCV baseline："
                f"ONNX={onnx_error}; HyperLPR3={hyperlpr_exc}"
            )
            return result

    result = recognize_with_opencv(image_bytes, provider=normalized, return_intermediate=return_intermediate)
    if normalized != "opencv_baseline":
        result.messages.append(f"未知 provider `{normalized}`，已使用 OpenCV baseline。")
    return result


def parse_ground_truth(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def collect_dataset_files(dataset_dir: str | None, sample_root: Path) -> list[Path]:
    if not dataset_dir:
        return []

    candidate = (sample_root / dataset_dir).resolve() if not Path(dataset_dir).is_absolute() else Path(dataset_dir).resolve()
    sample_root_resolved = sample_root.resolve()
    if not candidate.is_relative_to(sample_root_resolved):
        raise ValueError("dataset_dir 只能指向 data/samples 下的目录，避免误读系统文件。")
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError("dataset_dir 不存在或不是目录。")

    files: list[Path] = []
    for path in candidate.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)[:100]
