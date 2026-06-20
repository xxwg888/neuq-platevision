"""Post-processing rules for Chinese license-plate OCR outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .chars import PROVINCES

PROVINCE_SET = set(PROVINCES)
MAINLAND_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
LETTERS_WITH_IO = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")
SPECIAL_CHARS = set("警学港澳挂使领临")

CONFUSION_MAP = {
    "I": "1",
    "O": "0",
}


@dataclass(frozen=True)
class PlateRuleResult:
    text: str
    valid: bool
    reason: str


def normalize_plate_text(text: str) -> str:
    """Normalize common OCR ambiguities without changing Chinese province chars."""
    normalized = []
    for idx, ch in enumerate(text.strip().upper()):
        if idx >= 2:
            normalized.append(CONFUSION_MAP.get(ch, ch))
        else:
            normalized.append(ch)
    return "".join(normalized)


def expected_lengths(plate_type: str | None) -> set[int]:
    if plate_type == "green":
        return {8}
    if plate_type in {"blue", "yellow", "white", "black"}:
        return {7}
    return {7, 8}


def validate_china_plate(text: str, plate_type: str | None = None) -> PlateRuleResult:
    """Validate the coarse structural rules used in the report and demo output."""
    value = normalize_plate_text(text)
    if len(value) not in expected_lengths(plate_type):
        return PlateRuleResult(value, False, f"unexpected_length_{len(value)}")
    if not value:
        return PlateRuleResult(value, False, "empty")
    if value[0] not in PROVINCE_SET:
        return PlateRuleResult(value, False, "first_char_not_province")
    if len(value) >= 2 and value[1] not in MAINLAND_LETTERS:
        return PlateRuleResult(value, False, "second_char_not_mainland_letter")
    allowed_tail = MAINLAND_LETTERS | DIGITS | SPECIAL_CHARS
    invalid_tail = [ch for ch in value[2:] if ch not in allowed_tail]
    if invalid_tail:
        return PlateRuleResult(value, False, "tail_has_invalid_chars")
    return PlateRuleResult(value, True, "ok")


def infer_plate_type_from_length(text: str, fallback: str = "unknown") -> str:
    value = normalize_plate_text(text)
    if len(value) == 8:
        return "green"
    if len(value) == 7:
        return fallback if fallback != "unknown" else "blue"
    return fallback
