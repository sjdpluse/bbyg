from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectiveResult:
    threshold: float
    selected: int
    coverage: float
    accuracy: float | None
    balanced_accuracy: float | None
    logloss: float | None
    actual_long_fraction: float | None
    predicted_long_fraction: float | None
    minority_actual_count: int


def selective_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> SelectiveResult:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("aligned labels/probabilities required")
    if not 0.5 <= threshold < 1.0 or not np.isfinite(p).all():
        raise ValueError("invalid selective threshold or probabilities")
    confidence = np.maximum(p, 1.0 - p)
    mask = confidence >= threshold
    selected = int(mask.sum())
    coverage = float(selected / len(y))
    if selected == 0:
        return SelectiveResult(threshold, 0, coverage, None, None, None, None, None, 0)

    sy = y[mask]
    sp = np.clip(p[mask], 1e-9, 1.0 - 1e-9)
    pred = (sp >= 0.5).astype(int)
    counts = [int((sy == cls).sum()) for cls in (0, 1)]
    recalls: list[float] = []
    for cls in (0, 1):
        cls_mask = sy == cls
        if cls_mask.any():
            recalls.append(float(np.mean(pred[cls_mask] == cls)))
    balanced = float(np.mean(recalls)) if len(recalls) == 2 else None
    logloss = float(np.mean(-(sy * np.log(sp) + (1 - sy) * np.log(1 - sp))))
    return SelectiveResult(
        threshold=float(threshold),
        selected=selected,
        coverage=coverage,
        accuracy=float(np.mean(pred == sy)),
        balanced_accuracy=balanced,
        logloss=logloss,
        actual_long_fraction=float(np.mean(sy)),
        predicted_long_fraction=float(np.mean(pred)),
        minority_actual_count=min(counts),
    )


def choose_selective_threshold(
    y: np.ndarray,
    p: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60),
    min_coverage: float = 0.05,
    min_selected: int = 50,
    min_class_count: int = 20,
) -> tuple[float | None, list[dict]]:
    if not 0.0 < min_coverage <= 1.0 or min_selected < 1 or min_class_count < 1:
        raise ValueError("invalid selective gate settings")
    evaluated: list[dict] = []
    valid: list[SelectiveResult] = []
    for threshold in thresholds:
        result = selective_metrics(y, p, float(threshold))
        item = result.__dict__.copy()
        item["eligible"] = bool(
            result.coverage >= min_coverage
            and result.selected >= min_selected
            and result.minority_actual_count >= min_class_count
            and result.balanced_accuracy is not None
        )
        evaluated.append(item)
        if item["eligible"]:
            valid.append(result)
    if not valid:
        return None, evaluated
    best = max(
        valid,
        key=lambda r: (
            float(r.balanced_accuracy),
            float(r.accuracy),
            r.coverage,
            -r.threshold,
        ),
    )
    return best.threshold, evaluated
