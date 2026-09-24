from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json

import numpy as np

from .research_models import RobustScaler, binary_metrics, fit_logit, restore_training_prior
from .store import ScalperStore


NS = 1_000_000_000


def _day_bounds(day: str) -> tuple[int, int]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start.timestamp() + 86400
    return int(start.timestamp() * NS), int(end * NS)


@dataclass(frozen=True)
class EpisodeMatrix:
    ts_ns: np.ndarray
    x: np.ndarray
    regime: np.ndarray
    long_reward: np.ndarray
    short_reward: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return (self.long_reward > self.short_reward).astype(np.int8)

    @property
    def margin(self) -> np.ndarray:
        return self.long_reward - self.short_reward


@dataclass(frozen=True)
class BaselineSettings:
    train_days: tuple[str, ...] = (
        "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22"
    )
    calibration_day: str = "2026-09-23"
    diagnostic_day: str = "2026-09-24"
    minimum_training_margin: float = 0.10
    minimum_training_samples: int = 10_000
    memory_max_prototypes: int = 2048
    memory_k: int = 16
    linear_iterations: int = 180
    seed: int = 731022

    def __post_init__(self) -> None:
        if self.minimum_training_margin < 0:
            raise ValueError("minimum_training_margin must be non-negative")
        if self.minimum_training_samples < 100:
            raise ValueError("minimum_training_samples must be >=100")
        if self.memory_max_prototypes < 64 or self.memory_k < 1:
            raise ValueError("invalid memory settings")
        if self.memory_k > self.memory_max_prototypes:
            raise ValueError("memory_k cannot exceed prototype cap")


def load_days(store: ScalperStore, days: tuple[str, ...] | list[str]) -> EpisodeMatrix:
    clauses: list[str] = []
    args: list[int] = []
    for day in days:
        lo, hi = _day_bounds(day)
        clauses.append("(ts_ns>=? AND ts_ns<?)")
        args.extend((lo, hi))
    if not clauses:
        raise ValueError("at least one day required")
    rows = store.db.execute(
        "SELECT ts_ns,state_embedding_json,regime,counterfactual_long_reward,counterfactual_short_reward "
        "FROM market_episodes WHERE (" + " OR ".join(clauses) + ") ORDER BY ts_ns",
        args,
    ).fetchall()
    if not rows:
        raise ValueError(f"no v4 episodes for days: {days}")
    vectors = [json.loads(row[1]) for row in rows]
    width = len(vectors[0])
    if width == 0 or any(len(v) != width for v in vectors):
        raise ValueError("inconsistent episode embeddings")
    x = np.asarray(vectors, dtype=np.float64)
    long_reward = np.asarray([row[3] for row in rows], dtype=np.float64)
    short_reward = np.asarray([row[4] for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(long_reward).all() or not np.isfinite(short_reward).all():
        raise ValueError("non-finite episode matrix")
    return EpisodeMatrix(
        ts_ns=np.asarray([row[0] for row in rows], dtype=np.int64),
        x=x,
        regime=np.asarray([row[2] for row in rows], dtype=str),
        long_reward=long_reward,
        short_reward=short_reward,
    )


def _constant_prior(train_y: np.ndarray, n: int) -> np.ndarray:
    prior = float(np.clip(np.mean(train_y), 1e-6, 1 - 1e-6))
    return np.full(n, prior, dtype=np.float64)


def _regime_prior(train: EpisodeMatrix, train_mask: np.ndarray, target: EpisodeMatrix) -> np.ndarray:
    global_prior = float(np.clip(np.mean(train.y[train_mask]), 1e-6, 1 - 1e-6))
    mapping: dict[str, float] = {}
    for regime in np.unique(train.regime):
        mask = train_mask & (train.regime == regime)
        if int(mask.sum()) >= 50:
            mapping[str(regime)] = float(np.clip(np.mean(train.y[mask]), 1e-6, 1 - 1e-6))
    return np.asarray([mapping.get(str(r), global_prior) for r in target.regime], dtype=np.float64)


def _linear_probability(scaler, model, train_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    raw = model.probability(scaler.transform(target_x))
    return restore_training_prior(raw, train_y)


def _prototype_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, cap, dtype=np.int64)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _memory_probability(
    train: EpisodeMatrix,
    train_mask: np.ndarray,
    target: EpisodeMatrix,
    *,
    max_prototypes: int,
    k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    train_x = train.x[train_mask]
    train_y = train.y[train_mask].astype(np.float64)
    scaler = RobustScaler.fit(train_x)
    normalized_train = _normalize_rows(scaler.transform(train_x))
    ids = _prototype_indices(len(normalized_train), max_prototypes)
    prototypes = normalized_train[ids]
    labels = train_y[ids]
    out = np.empty(len(target.x), dtype=np.float64)
    target_scaled = scaler.transform(target.x)
    for begin in range(0, len(target_scaled), batch_size):
        chunk = _normalize_rows(target_scaled[begin:begin + batch_size])
        sim = chunk @ prototypes.T
        kk = min(k, prototypes.shape[0])
        nearest = np.argpartition(sim, -kk, axis=1)[:, -kk:]
        nearest_sim = np.take_along_axis(sim, nearest, axis=1)
        nearest_y = labels[nearest]
        weights = np.square(np.maximum(nearest_sim, 0.0)) + 1e-6
        out[begin:begin + len(chunk)] = np.sum(weights * nearest_y, axis=1) / np.sum(weights, axis=1)
    return np.clip(out, 1e-6, 1 - 1e-6)


def _economic_proxy(data: EpisodeMatrix, p: np.ndarray) -> dict:
    side_long = p >= 0.5
    selected = np.where(side_long, data.long_reward, data.short_reward)
    confidence = np.maximum(p, 1.0 - p)
    report: dict[str, object] = {
        "mean_selected_reward": float(np.mean(selected)),
        "median_selected_reward": float(np.median(selected)),
        "positive_selected_reward_fraction": float(np.mean(selected > 0)),
        "long_selection_fraction": float(np.mean(side_long)),
    }
    for threshold in (0.55, 0.60):
        mask = confidence >= threshold
        key = f"confidence_{int(threshold * 100)}"
        report[key] = {
            "coverage": float(np.mean(mask)),
            "count": int(mask.sum()),
            "mean_selected_reward": None if not mask.any() else float(np.mean(selected[mask])),
            "positive_reward_fraction": None if not mask.any() else float(np.mean(selected[mask] > 0)),
        }
    return report


def _score(data: EpisodeMatrix, p: np.ndarray) -> dict:
    return {
        "classification": binary_metrics(data.y, p),
        "economic_proxy": _economic_proxy(data, p),
    }


def run_v4_baselines(store: ScalperStore, settings: BaselineSettings | None = None) -> dict:
    settings = settings or BaselineSettings()
    train = load_days(store, settings.train_days)
    calibration = load_days(store, [settings.calibration_day])
    diagnostic = load_days(store, [settings.diagnostic_day])
    train_mask = np.abs(train.margin) >= settings.minimum_training_margin
    if int(train_mask.sum()) < settings.minimum_training_samples:
        raise ValueError(f"insufficient strong-margin training episodes: {int(train_mask.sum())}")
    train_y = train.y[train_mask]

    scaler = RobustScaler.fit(train.x[train_mask])
    linear_model = fit_logit(
        scaler.transform(train.x[train_mask]), train_y,
        iterations=settings.linear_iterations, balanced=True,
    )

    def probabilities(data: EpisodeMatrix) -> dict[str, np.ndarray]:
        constant = _constant_prior(train_y, len(data.x))
        regime = _regime_prior(train, train_mask, data)
        linear = _linear_probability(scaler, linear_model, train_y, data.x)
        memory = _memory_probability(
            train, train_mask, data,
            max_prototypes=settings.memory_max_prototypes,
            k=settings.memory_k,
        )
        combo = np.clip(0.5 * linear + 0.5 * memory, 1e-6, 1 - 1e-6)
        return {
            "constant_prior": constant,
            "regime_prior": regime,
            "linear_logit": linear,
            "episodic_memory": memory,
            "linear_plus_memory": combo,
        }

    calibration_p = probabilities(calibration)
    diagnostic_p = probabilities(diagnostic)
    return {
        "protocol": "bbyg_v4_research_v1",
        "execution_authorized": False,
        "settings": {
            "train_days": settings.train_days,
            "calibration_day": settings.calibration_day,
            "diagnostic_day": settings.diagnostic_day,
            "minimum_training_margin": settings.minimum_training_margin,
            "minimum_training_samples": settings.minimum_training_samples,
            "memory_max_prototypes": settings.memory_max_prototypes,
            "memory_k": settings.memory_k,
            "linear_iterations": settings.linear_iterations,
        },
        "counts": {
            "train_all": int(len(train.x)),
            "train_strong_margin": int(train_mask.sum()),
            "calibration": int(len(calibration.x)),
            "diagnostic": int(len(diagnostic.x)),
        },
        "train_target": {
            "long_fraction": float(np.mean(train_y)),
            "mean_abs_margin": float(np.mean(np.abs(train.margin[train_mask]))),
        },
        "calibration": {name: _score(calibration, p) for name, p in calibration_p.items()},
        "diagnostic_only": {name: _score(diagnostic, p) for name, p in diagnostic_p.items()},
        "selection_rule": (
            "Choose/freeze the candidate family using calibration only. "
            "The diagnostic-only day is reported for debugging and must not be used for model selection."
        ),
        "pristine_validation": {
            "status": "waiting_for_future_data",
            "strictly_after_utc_date": "2026-09-24",
            "minimum_full_days": 2,
            "minimum_total_episodes": 40000,
        },
    }
