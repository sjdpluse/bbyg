from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Iterable

import numpy as np

from .store import ScalperStore


@dataclass(frozen=True)
class SplitPlan:
    train_end_ts_ns: int
    calibration_start_ts_ns: int
    calibration_end_ts_ns: int
    validation_start_ts_ns: int
    validation_end_ts_ns: int
    train_count: int
    calibration_count: int
    validation_count: int


@dataclass(frozen=True)
class DatasetDiagnostics:
    episode_count: int
    feature_dimensions: int
    day_count: int
    regime_counts: dict[str, int]
    regime_fractions: dict[str, float]
    reward: dict[str, dict]
    directional_edge: dict
    per_day: list[dict]
    pathologies: dict
    split_plan: SplitPlan | None
    gates: dict


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "p01": None, "p05": None, "p25": None, "p50": None,
                "p75": None, "p95": None, "p99": None, "max": None, "mean": None, "std": None}
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(np.min(values)), "p01": float(q[0]), "p05": float(q[1]),
        "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]),
        "p95": float(q[5]), "p99": float(q[6]), "max": float(np.max(values)),
        "mean": float(np.mean(values)), "std": float(np.std(values)),
    }


def _utc_day(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


class V4DatasetDiagnostics:
    """Read-only diagnostics over persisted policy-independent market episodes."""

    def __init__(self, *, min_day_episodes: int = 2000, minimum_validation_days: int = 1,
                 minimum_calibration_days: int = 1):
        self.min_day_episodes = int(min_day_episodes)
        self.minimum_validation_days = int(minimum_validation_days)
        self.minimum_calibration_days = int(minimum_calibration_days)
        if self.min_day_episodes < 100:
            raise ValueError("min_day_episodes too small")

    def analyze(self, store: ScalperStore) -> DatasetDiagnostics:
        rows = store.db.execute(
            """SELECT ts_ns,state_embedding_json,regime,
                      counterfactual_long_reward,counterfactual_short_reward,context_json
               FROM market_episodes ORDER BY ts_ns"""
        ).fetchall()
        if not rows:
            raise ValueError("market_episodes is empty")

        ts = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
        embeddings = [json.loads(r[1]) for r in rows]
        dims = {len(v) for v in embeddings}
        if len(dims) != 1:
            raise ValueError("mixed feature dimensions in market_episodes")
        feature_dimensions = int(next(iter(dims)))
        regimes = [str(r[2]) for r in rows]
        long_reward = np.asarray([float(r[3]) for r in rows], dtype=float)
        short_reward = np.asarray([float(r[4]) for r in rows], dtype=float)
        if not np.isfinite(long_reward).all() or not np.isfinite(short_reward).all():
            raise ValueError("non-finite counterfactual rewards")

        regime_counts: dict[str, int] = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        n = len(rows)
        regime_fractions = {k: v / n for k, v in sorted(regime_counts.items())}

        reward = {
            "long": _quantiles(long_reward),
            "short": _quantiles(short_reward),
            "best_side": _quantiles(np.maximum(long_reward, short_reward)),
            "long_minus_short": _quantiles(long_reward - short_reward),
        }
        long_better = long_reward > short_reward
        ties = np.isclose(long_reward, short_reward, atol=1e-12, rtol=0.0)
        directional_edge = {
            "long_better_fraction": float(np.mean(long_better & ~ties)),
            "short_better_fraction": float(np.mean((~long_better) & ~ties)),
            "tie_fraction": float(np.mean(ties)),
            "mean_long_minus_short": float(np.mean(long_reward - short_reward)),
            "median_abs_directional_margin": float(np.median(np.abs(long_reward - short_reward))),
            "strong_direction_fraction_abs_ge_0p10": float(np.mean(np.abs(long_reward - short_reward) >= 0.10)),
        }

        by_day: dict[str, list[int]] = {}
        for i, stamp in enumerate(ts.tolist()):
            by_day.setdefault(_utc_day(stamp), []).append(i)
        per_day: list[dict] = []
        qualified_days: list[str] = []
        for day in sorted(by_day):
            ids = np.asarray(by_day[day], dtype=int)
            lr = long_reward[ids]
            sr = short_reward[ids]
            day_regimes: dict[str, int] = {}
            for i in ids.tolist():
                day_regimes[regimes[i]] = day_regimes.get(regimes[i], 0) + 1
            record = {
                "utc_date": day,
                "episodes": int(len(ids)),
                "min_ts_ns": int(ts[ids[0]]),
                "max_ts_ns": int(ts[ids[-1]]),
                "mean_long_reward": float(np.mean(lr)),
                "mean_short_reward": float(np.mean(sr)),
                "mean_best_reward": float(np.mean(np.maximum(lr, sr))),
                "mean_directional_margin": float(np.mean(lr - sr)),
                "strong_direction_fraction_abs_ge_0p10": float(np.mean(np.abs(lr - sr) >= 0.10)),
                "regimes": dict(sorted(day_regimes.items())),
                "eligible_for_split": bool(len(ids) >= self.min_day_episodes),
            }
            if record["eligible_for_split"]:
                qualified_days.append(day)
            per_day.append(record)

        # Chronological day-level split only. Never split inside a day after inspecting rewards.
        split_plan: SplitPlan | None = None
        needed = self.minimum_calibration_days + self.minimum_validation_days + 1
        if len(qualified_days) >= needed:
            validation_days = qualified_days[-self.minimum_validation_days:]
            calibration_end_index = len(qualified_days) - self.minimum_validation_days
            calibration_days = qualified_days[
                calibration_end_index - self.minimum_calibration_days:calibration_end_index
            ]
            train_days = qualified_days[:calibration_end_index - self.minimum_calibration_days]
            train_ids = np.asarray([i for d in train_days for i in by_day[d]], dtype=int)
            cal_ids = np.asarray([i for d in calibration_days for i in by_day[d]], dtype=int)
            val_ids = np.asarray([i for d in validation_days for i in by_day[d]], dtype=int)
            split_plan = SplitPlan(
                train_end_ts_ns=int(ts[train_ids[-1]]),
                calibration_start_ts_ns=int(ts[cal_ids[0]]),
                calibration_end_ts_ns=int(ts[cal_ids[-1]]),
                validation_start_ts_ns=int(ts[val_ids[0]]),
                validation_end_ts_ns=int(ts[val_ids[-1]]),
                train_count=int(len(train_ids)),
                calibration_count=int(len(cal_ids)),
                validation_count=int(len(val_ids)),
            )

        contexts = [json.loads(r[5]) for r in rows]
        net_long = np.asarray([float(c.get("long_net_r", math.nan)) for c in contexts], dtype=float)
        net_short = np.asarray([float(c.get("short_net_r", math.nan)) for c in contexts], dtype=float)
        finite_net = np.isfinite(net_long) & np.isfinite(net_short)
        pathologies = {
            "reward_at_bound_fraction": float(np.mean(
                (np.abs(long_reward) >= 0.999999) | (np.abs(short_reward) >= 0.999999)
            )),
            "net_r_available_fraction": float(np.mean(finite_net)),
            "extreme_net_r_fraction_abs_gt_10": (
                None if not np.any(finite_net)
                else float(np.mean((np.abs(net_long[finite_net]) > 10) | (np.abs(net_short[finite_net]) > 10)))
            ),
            "missing_quiet_regime": "quiet" not in regime_counts,
        }

        largest_regime_fraction = max(regime_fractions.values())
        gates = {
            "episode_count_ok": n >= 50_000,
            "feature_schema_ok": feature_dimensions >= 32,
            "reward_finite_ok": True,
            "reward_not_saturated_ok": pathologies["reward_at_bound_fraction"] < 0.05,
            "regime_concentration_ok": largest_regime_fraction < 0.80,
            "split_possible": split_plan is not None,
            "direction_not_collapsed_ok": max(
                directional_edge["long_better_fraction"], directional_edge["short_better_fraction"]
            ) < 0.75,
        }
        gates["all_training_gates_pass"] = all(gates.values())
        return DatasetDiagnostics(
            episode_count=n,
            feature_dimensions=feature_dimensions,
            day_count=len(per_day),
            regime_counts=dict(sorted(regime_counts.items())),
            regime_fractions=regime_fractions,
            reward=reward,
            directional_edge=directional_edge,
            per_day=per_day,
            pathologies=pathologies,
            split_plan=split_plan,
            gates=gates,
        )
