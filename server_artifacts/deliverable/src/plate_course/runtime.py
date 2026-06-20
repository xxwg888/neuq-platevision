"""Runtime helpers shared by training, evaluation, export, and demos."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .chars import CHARSET
from .model import build_recognizer


def get_checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("args", {}))


def get_checkpoint_model_name(checkpoint: dict[str, Any], default: str = "crnn_lite") -> str:
    args = get_checkpoint_args(checkpoint)
    config = checkpoint.get("config", {})
    return str(args.get("model") or config.get("model_name") or config.get("model") or default)


def get_checkpoint_image_size(checkpoint: dict[str, Any]) -> tuple[int, int]:
    args = get_checkpoint_args(checkpoint)
    return int(args.get("image_height", 48)), int(args.get("image_width", 160))


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device | str = "cpu") -> nn.Module:
    args = get_checkpoint_args(checkpoint)
    model_name = get_checkpoint_model_name(checkpoint)
    hidden_size = int(args.get("hidden_size", 128))
    num_layers = int(args.get("num_layers", 1))
    model = build_recognizer(model_name, num_classes=len(CHARSET), hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
