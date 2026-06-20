"""Lightweight CTC models for cropped license plate recognition."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRNNLite(nn.Module):
    """A small CNN + BiGRU CTC recognizer.

    Input: RGB plate crop tensor, N x 3 x 48 x 160.
    Output: unnormalized logits, N x T x C. For the default width, T is 40.
    """

    def __init__(self, num_classes: int, input_channels: int = 3, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24 x 80
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12 x 40
            ConvBlock(64, 96),
            ConvBlock(96, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 6 x 40
            ConvBlock(128, 160),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 3 x 40
            ConvBlock(160, 192),
            nn.AvgPool2d(kernel_size=(3, 1), stride=(1, 1)),  # 1 x 40
        )
        self.sequence = nn.GRU(
            input_size=192,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(2).permute(0, 2, 1).contiguous()
        x, _ = self.sequence(x)
        return self.classifier(x)


class SmallBasicBlock(nn.Module):
    """Small residual-style block used by LPRNet-lite."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class LPRNetLite(nn.Module):
    """A compact LPRNet-style CTC recognizer for full plate crops.

    Input: RGB plate crop tensor, N x 3 x 48 x 160.
    Output: unnormalized logits, N x T x C. T follows the final feature width.
    """

    def __init__(self, num_classes: int, input_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),  # 48 x 80
            SmallBasicBlock(64),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),  # 24 x 40
            SmallBasicBlock(128),
            SmallBasicBlock(128),
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),  # 12 x 40
            nn.Dropout(dropout),
            ConvBlock(256, 256),
            SmallBasicBlock(256),
            nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, None)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x.squeeze(2).permute(0, 2, 1).contiguous()


def build_recognizer(name: str, num_classes: int, hidden_size: int = 128,
                     num_layers: int = 1) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"crnn", "crnn_lite", "crnnlite"}:
        return CRNNLite(num_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers)
    if normalized in {"lprnet", "lprnet_lite", "lprnetlite"}:
        return LPRNetLite(num_classes=num_classes)
    raise ValueError(f"Unknown recognizer model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
