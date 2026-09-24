from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from truetrade.scalper.research_models import (
    binary_metrics,
    fit_predict_architecture,
    restore_training_prior,
)
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.v4_baselines import EpisodeMatrix, load_days


ARCHITECTURES = ("linear", "quadratic", "mlp")
CONFIDENCE_THRESHOLDS = (0.55, 0.60)


@dataclass(frozen=True)
class NonlinearSettings:
    train_days: tuple[str, ...] = (
        "2026-09-16",
        "2026-09-17",
        "2026-09-18",
        "2026-09-21",
        "2026-09-22",
    )
    calibration_day: str = "2026-09-23"
    diagnostic_day: str = "2026-09-24"
    minimum_training_margin: float = 0.10
    minimum_training_samples: int = 10_000
    minimum_day_samples: int = 3_000
    recent_training_samples: int = 20_000
    linear_iterations: int = 140
    mlp_iterations: int = 120
    seed: int = 731022

    def __post_init__(self) -> None:
        if self.minimum_training_margin < 0:
            raise ValueError("minimum_training_margin must be non-negative")
        if self.minimum_training_samples < 100:
            raise ValueError("minimum_training_samples must be >= 100")
        if self.minimum_day_samples < 100:
            raise ValueError("minimum_day_samples must be >= 100")
        if self.recent_training_samples < self.minimum_training_samples:
            raise ValueError("recent_training_samples cannot be smaller than minimum_training_samples")
        if self.linear_iterations < 1 or self.mlp_iterations < 1:
            raise ValueError("optimizer iterations must be positive")


def _constant_prior(train_y: np.ndarray, n: int) -> np.ndarray:
    prior = float(np.clip(np.mean(train_y), 1e-9, 1.0 - 1e-9))
    return np.full(n, prior, dtype=np.float64)


def _economic_proxy(data: EpisodeMatrix, p: np.ndarray) -> dict[str, object]:
    p = np.asarray(p, dtype=np.float64)
    side_long = p >= 0.5
    selected = np.where(side_long, data.long_reward, data.short_reward)
    confidence = np.maximum(p, 1.0 - p)
    result: dict[str, object] = {
        "mean_selected_reward": float(np.mean(selected)),
        "median_selected_reward": float(np.median(selected)),
        "positive_selected_reward_fraction": float(np.mean(selected > 0)),
        "long_selection_fraction": float(np.mean(side_long)),
    }
    for threshold in CONFIDENCE_THRESHOLDS:
        mask = confidence >= threshold
        result[f"confidence_{int(threshold * 100)}"] = {
            "coverage": float(np.mean(mask)),
            "count": int(mask.sum()),
            "mean_selected_reward": None if not mask.any() else float(np.mean(selected[mask])),
            "positive_reward_fraction": None if not mask.any() else float(np.mean(selected[mask] > 0)),
        }
    return result


def _score(data: EpisodeMatrix, p: np.ndarray) -> dict[str, object]:
    return {
        "classification": binary_metrics(data.y, p),
        "economic_proxy": _economic_proxy(data, p),
    }


def _validate_day(name: str, data: EpisodeMatrix, minimum_day_samples: int) -> None:
    if len(data.x) < minimum_day_samples:
        raise ValueError(
            f"{name} has only {len(data.x)} episodes; minimum required is {minimum_day_samples}"
        )


def _candidate_probabilities(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    *,
    settings: NonlinearSettings,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    raw = fit_predict_architecture(
        architecture,
        train_x,
        train_y,
        target_x,
        linear_iterations=settings.linear_iterations,
        mlp_iterations=settings.mlp_iterations,
        seed=settings.seed + seed_offset,
        restore_prior=False,
    )
    corrected = restore_training_prior(raw, train_y)
    return {
        "raw_balanced": np.clip(raw, 1e-9, 1.0 - 1e-9),
        "prior_corrected": np.clip(corrected, 1e-9, 1.0 - 1e-9),
    }


def _evaluate_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target: EpisodeMatrix,
    *,
    settings: NonlinearSettings,
    seed_offset: int,
) -> dict[str, object]:
    baseline_p = _constant_prior(train_y, len(target.x))
    baseline = _score(target, baseline_p)
    candidates: dict[str, object] = {}
    for architecture in ARCHITECTURES:
        variants = _candidate_probabilities(
            architecture,
            train_x,
            train_y,
            target.x,
            settings=settings,
            seed_offset=seed_offset,
        )
        scored: dict[str, object] = {}
        for variant, p in variants.items():
            metrics = _score(target, p)
            metrics["classification"]["logloss_improvement_vs_constant_prior"] = float(
                baseline["classification"]["logloss"] - metrics["classification"]["logloss"]
            )
            scored[variant] = metrics
        candidates[architecture] = scored
    return {"constant_prior": baseline, "candidates": candidates}


def _calibration_order(calibration: dict[str, object]) -> list[str]:
    candidates = calibration["candidates"]
    return sorted(
        ARCHITECTURES,
        key=lambda name: (
            -float(
                candidates[name]["prior_corrected"]["classification"][
                    "logloss_improvement_vs_constant_prior"
                ]
            ),
            -float(candidates[name]["prior_corrected"]["classification"]["balanced_accuracy"]),
            -float(candidates[name]["prior_corrected"]["economic_proxy"]["mean_selected_reward"]),
        ),
    )


def run(settings: NonlinearSettings) -> dict[str, object]:
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    db_path = state_dir / "scalper.sqlite"
    if not db_path.exists():
        raise ValueError(f"BBYG store not found: {db_path}")

    with ScalperStore(db_path) as store:
        train = load_days(store, settings.train_days)
        calibration = load_days(store, [settings.calibration_day])
        diagnostic = load_days(store, [settings.diagnostic_day])

    _validate_day("calibration", calibration, settings.minimum_day_samples)
    _validate_day("diagnostic", diagnostic, settings.minimum_day_samples)

    strong_mask = np.abs(train.margin) >= settings.minimum_training_margin
    strong_indices = np.flatnonzero(strong_mask)
    if len(strong_indices) < settings.minimum_training_samples:
        raise ValueError(
            f"insufficient strong-margin training episodes: {len(strong_indices)}"
        )
    if len(strong_indices) > settings.recent_training_samples:
        strong_indices = strong_indices[-settings.recent_training_samples :]

    train_x = train.x[strong_indices]
    train_y = train.y[strong_indices]
    calibration_result = _evaluate_split(
        train_x,
        train_y,
        calibration,
        settings=settings,
        seed_offset=1,
    )
    diagnostic_result = _evaluate_split(
        train_x,
        train_y,
        diagnostic,
        settings=settings,
        seed_offset=2,
    )
    calibration_order = _calibration_order(calibration_result)

    return {
        "stage": "bbyg_v4_nonlinear_benchmark",
        "protocol": "bbyg_v4_nonlinear_research_v1",
        "mode": "read_only_day_isolated_research",
        "execution_authorized": False,
        "research_config": {
            "train_days": settings.train_days,
            "calibration_day": settings.calibration_day,
            "diagnostic_day": settings.diagnostic_day,
            "minimum_training_margin": settings.minimum_training_margin,
            "minimum_training_samples": settings.minimum_training_samples,
            "minimum_day_samples": settings.minimum_day_samples,
            "recent_training_samples": settings.recent_training_samples,
            "linear_iterations": settings.linear_iterations,
            "mlp_iterations": settings.mlp_iterations,
            "seed": settings.seed,
            "architectures": ARCHITECTURES,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        },
        "counts": {
            "train_all": int(len(train.x)),
            "train_strong_margin_all": int(strong_mask.sum()),
            "train_used": int(len(train_x)),
            "calibration": int(len(calibration.x)),
            "diagnostic": int(len(diagnostic.x)),
        },
        "train_target": {
            "long_fraction": float(np.mean(train_y)),
            "mean_absolute_reward_margin": float(
                np.mean(np.abs(train.long_reward[strong_indices] - train.short_reward[strong_indices]))
            ),
        },
        "calibration": calibration_result,
        "calibration_research_order": calibration_order,
        "diagnostic_only": diagnostic_result,
        "selection_rule": (
            "Model-family ordering is frozen from calibration only. Diagnostic-only results are "
            "reported for debugging and must not change the selected family or thresholds."
        ),
        "pristine_validation": {
            "status": "waiting_for_future_data",
            "strictly_after_utc_date": "2026-09-24",
            "minimum_full_eligible_days": 2,
            "minimum_total_episodes": 40000,
            "peek_before_freeze_forbidden": True,
        },
        "research_constraints": [
            "No MT5 execution is authorized by this benchmark.",
            "No validation labels are used during model fitting.",
            "The diagnostic-only day cannot be used for family selection or threshold tuning.",
            "A nonlinear candidate must still beat simpler matched baselines on untouched future data before promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only nonlinear benchmark over frozen BBYG v4 market episodes"
    )
    parser.add_argument("--min-train", type=int, default=10_000)
    parser.add_argument("--min-day-samples", type=int, default=3_000)
    parser.add_argument("--recent", type=int, default=20_000)
    parser.add_argument("--linear-iterations", type=int, default=140)
    parser.add_argument("--mlp-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()

    settings = NonlinearSettings(
        minimum_training_samples=args.min_train,
        minimum_day_samples=args.min_day_samples,
        recent_training_samples=args.recent,
        linear_iterations=args.linear_iterations,
        mlp_iterations=args.mlp_iterations,
        seed=args.seed,
    )
    print(json.dumps(run(settings), sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
