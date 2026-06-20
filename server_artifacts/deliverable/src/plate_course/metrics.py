"""Evaluation metrics for plate OCR."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


def recognition_metrics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """Return plate-level and character-level recognition metrics."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    total = max(len(targets), 1)
    exact = sum(int(p == t) for p, t in zip(predictions, targets))
    edits = [edit_distance(p, t) for p, t in zip(predictions, targets)]
    target_chars = sum(len(t) for t in targets)
    correct_chars = sum(max(len(t) - d, 0) for d, t in zip(edits, targets))
    char_acc = correct_chars / max(target_chars, 1)
    return {
        "plate_accuracy": exact / total,
        "character_accuracy": char_acc,
        "avg_edit_distance": sum(edits) / total,
    }

