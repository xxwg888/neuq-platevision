"""Character tables for Chinese license plate recognition."""

from __future__ import annotations

BLANK_TOKEN = "<blank>"

PROVINCES = list(
    "京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳"
)

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
SPECIALS = list("警学挂领使临")

CHARSET = [BLANK_TOKEN] + PROVINCES + LETTERS + DIGITS + SPECIALS

CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARSET)}
IDX_TO_CHAR = {i: ch for ch, i in CHAR_TO_IDX.items()}

CCPD_PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
]

CCPD_ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


def encode_label(text: str) -> list[int]:
    """Encode a plate string to CTC target ids."""
    unknown = [ch for ch in text if ch not in CHAR_TO_IDX]
    if unknown:
        raise ValueError(f"Unsupported plate characters: {unknown!r}")
    return [CHAR_TO_IDX[ch] for ch in text]


def decode_ccpd_plate(indices: list[int]) -> str:
    """Decode CCPD filename label indices to plate text."""
    if len(indices) < 2:
        raise ValueError(f"CCPD plate label needs at least two indices: {indices}")
    chars = [CCPD_PROVINCES[indices[0]], CCPD_ADS[indices[1]]]
    chars.extend(CCPD_ADS[i] for i in indices[2:])
    return "".join(chars)


def decode_indices(indices: list[int], collapse_repeats: bool = True) -> str:
    """Decode CTC indices to a plate string."""
    chars: list[str] = []
    prev = None
    for idx in indices:
        if idx == 0:
            prev = idx
            continue
        if collapse_repeats and idx == prev:
            continue
        chars.append(IDX_TO_CHAR.get(int(idx), ""))
        prev = idx
    return "".join(chars)


def greedy_decode(logits) -> list[str]:
    """Greedy CTC decode for logits shaped as N x T x C."""
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    return [decode_indices(row) for row in pred]
