from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
EVALUATION_DIR = DATA_DIR / "evaluations"
SAMPLES_DIR = DATA_DIR / "samples"
REMOTE_SETTINGS_FILE = DATA_DIR / "remote_settings.json"

for directory in (UPLOAD_DIR, OUTPUT_DIR, EVALUATION_DIR, SAMPLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class RemoteSettings(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    timeout_seconds: float = Field(default=12.0, ge=1.0, le=120.0)


def load_remote_settings() -> RemoteSettings:
    if not REMOTE_SETTINGS_FILE.exists():
        return RemoteSettings()
    try:
        raw: dict[str, Any] = json.loads(REMOTE_SETTINGS_FILE.read_text(encoding="utf-8"))
        return RemoteSettings(**raw)
    except (OSError, ValueError, TypeError):
        return RemoteSettings()


def save_remote_settings(settings: RemoteSettings) -> None:
    REMOTE_SETTINGS_FILE.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )

