from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .research_models import RobustScaler, binary_metrics, fit_logit
from .store import ScalperStore
from .types import MicroFeatures, Side, Tick


@dataclass(frozen=True)
class TechnicalSnapshot:
    ready: bool
    long_score: int
    short_score: int
    rsi: float | None
    ema_fast: float | None
    ema_slow: float | None
    atr: float | None
    atr_spreads: float | None
    regime: str

    def score_for(self, side: Side) -> int:
        return self.long_score if side is Side.LONG else self.short_score

    def opposite_score(self, side: Side) -> int:
        return self.short_score if side is Side.LONG else self.long_score


@dataclass
class _Bar:
    second: int
    open: float
    high: float
    low: float
    close: float


class TechnicalAnalyzer:
    """Small causal 1-second technical context for XAUUSD.

    This is deliberately separate from the learned microstructure classifier. It acts as
    an independent confirmation layer so a weak probability wobble around 0.50 cannot be
    treated as a full trading thesis by itself.
    """

    def __init__(self, max_bars: int = 240):
        self.bars: deque[_Bar] = deque(maxlen=max_bars)
        self.current: _Bar | None = None

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> float:
        alpha = 2.0 / (period + 1.0)
        value = float(values[0])
        for x in values[1:]:
            value = alpha * float(x) + (1.0 - alpha) * value
        return value

    @staticmethod
    def _rsi(values: np.ndarray, period: int = 14) -> float:
        diffs = np.diff(values[-(period + 1):])
        if len(diffs) < period:
            return 50.0
        gains = np.maximum(diffs, 0.0)
        losses = np.maximum(-diffs, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss <= 1e-12:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _atr(bars: list[_Bar], period: int = 14) -> float:
        if len(bars) < period + 1:
            return 0.0
        trs: list[float] = []
        recent = bars[-(period + 1):]
        for prev, cur in zip(recent, recent[1:]):
            trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        return float(np.mean(trs)) if trs else 0.0

    def update(self, tick: Tick) -> TechnicalSnapshot:
        second = tick.ts_ns // 1_000_000_000
        mid = tick.mid
        if self.current is None:
            self.current = _Bar(second, mid, mid, mid, mid)
        elif second == self.current.second:
            self.current.high = max(self.current.high, mid)
            self.current.low = min(self.current.low, mid)
            self.current.close = mid
        else:
            self.bars.append(self.current)
            self.current = _Bar(second, mid, mid, mid, mid)

        all_bars = list(self.bars)
        if self.current is not None:
            all_bars.append(self.current)
        if len(all_bars) < 30:
            return TechnicalSnapshot(False, 0, 0, None, None, None, None, None, "warming")

        closes = np.asarray([b.close for b in all_bars], dtype=float)
        fast = self._ema(closes[-60:], 9)
        slow = self._ema(closes[-90:], 21)
        prior_fast = self._ema(closes[-61:-1], 9) if len(closes) >= 61 else fast
        rsi = self._rsi(closes, 14)
        atr = self._atr(all_bars, 14)
        atr_spreads = atr / max(tick.spread, 1e-12)
        recent = all_bars[-10:]
        structure_mid = (max(b.high for b in recent) + min(b.low for b in recent)) / 2.0
        close = closes[-1]

        long_score = 0
        short_score = 0
        if fast > slow:
            long_score += 1
        elif fast < slow:
            short_score += 1
        if fast > prior_fast:
            long_score += 1
        elif fast < prior_fast:
            short_score += 1
        if rsi >= 52.0:
            long_score += 1
        if rsi <= 48.0:
            short_score += 1
        if close > fast and close > structure_mid:
            long_score += 1
        if close < fast and close < structure_mid:
            short_score += 1

        trend_gap = abs(fast - slow)
        if atr <= 1e-12 or atr_spreads < 0.75:
            regime = "quiet"
        elif trend_gap >= 0.20 * atr:
            regime = "trend"
        else:
            regime = "chop"
        return TechnicalSnapshot(True, long_score, short_score, rsi, fast, slow, atr, atr_spreads, regime)


@dataclass(frozen=True)
class FundamentalSnapshot:
    active: bool
    mode: str
    confidence: float
    reason: str
    age_seconds: float | None


class FundamentalGate:
    """Hot-reloadable external fundamental/news context.

    Expected JSON example:
      {"updated_at_utc":"2026-09-24T10:00:00Z","mode":"neutral",
       "confidence":0.7,"reason":"no high-impact event"}

    The runner never invents fundamentals. Missing/stale data is reported as inactive.
    """

    VALID = {"neutral", "long", "short", "block"}

    def __init__(self, path: Path, *, max_age_seconds: float = 900.0):
        self.path = Path(path)
        self.max_age_seconds = float(max_age_seconds)
        self._last_read = 0.0
        self._cached = FundamentalSnapshot(False, "neutral", 0.0, "no_feed", None)

    def snapshot(self) -> FundamentalSnapshot:
        now_mono = time.monotonic()
        if now_mono - self._last_read < 2.0:
            return self._cached
        self._last_read = now_mono
        if not self.path.exists():
            self._cached = FundamentalSnapshot(False, "neutral", 0.0, "no_feed", None)
            return self._cached
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            raw = str(doc["updated_at_utc"]).replace("Z", "+00:00")
            updated = datetime.fromisoformat(raw)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = max(0.0, (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds())
            mode = str(doc.get("mode", "neutral")).lower()
            confidence = float(doc.get("confidence", 0.0))
            reason = str(doc.get("reason", "external_feed"))
            if mode not in self.VALID or not 0.0 <= confidence <= 1.0:
                raise ValueError("invalid fundamental context")
            if age > self.max_age_seconds:
                self._cached = FundamentalSnapshot(False, "neutral", confidence, "stale_feed", age)
            else:
                self._cached = FundamentalSnapshot(True, mode, confidence, reason, age)
        except Exception:
            self._cached = FundamentalSnapshot(False, "neutral", 0.0, "invalid_feed", None)
        return self._cached

    @staticmethod
    def allows(snapshot: FundamentalSnapshot, side: Side, *, require_active: bool = False) -> bool:
        if not snapshot.active:
            return not require_active
        if snapshot.mode == "block":
            return False
        if snapshot.mode == "neutral":
            return True
        return (snapshot.mode == "long" and side is Side.LONG) or (
            snapshot.mode == "short" and side is Side.SHORT
        )


@dataclass
class _PendingLabel:
    feature_ts_ns: int
    x: tuple[float, ...]
    entry: Tick | None = None
    long_alive: bool = True
    short_alive: bool = True
    long_profit: float = 0.0
    short_profit: float = 0.0
    long_stop: float = 0.0
    short_stop: float = 0.0
    ticks_observed: int = 0


@dataclass(frozen=True)
class ResolvedLiveLabel:
    feature_ts_ns: int
    label_end_ts_ns: int
    x: tuple[float, ...]
    y: int


class LiveLabelQueue:
    """Incremental executable first-passage labels for live learning."""

    def __init__(self, *, profit_spreads: float = 1.6, loss_spreads: float = 1.4,
                 extra_cost_spreads: float = 0.20, max_lookahead_ticks: int = 600):
        self.profit_spreads = float(profit_spreads)
        self.loss_spreads = float(loss_spreads)
        self.extra_cost_spreads = float(extra_cost_spreads)
        self.max_lookahead_ticks = int(max_lookahead_ticks)
        self.pending: deque[_PendingLabel] = deque()
        self.resolved = 0
        self.discarded = 0

    def add_decision(self, features: MicroFeatures) -> None:
        self.pending.append(_PendingLabel(features.ts_ns, features.vector()))

    def advance(self, tick: Tick) -> list[ResolvedLiveLabel]:
        resolved: list[ResolvedLiveLabel] = []
        keep: deque[_PendingLabel] = deque()
        while self.pending:
            item = self.pending.popleft()
            if item.entry is None:
                if tick.ts_ns <= item.feature_ts_ns:
                    keep.append(item)
                    continue
                item.entry = tick
                spread = max(tick.spread, 1e-12)
                item.long_profit = tick.ask + (self.profit_spreads + self.extra_cost_spreads) * spread
                item.short_profit = tick.bid - (self.profit_spreads + self.extra_cost_spreads) * spread
                item.long_stop = tick.ask - self.loss_spreads * spread
                item.short_stop = tick.bid + self.loss_spreads * spread
                keep.append(item)
                continue

            item.ticks_observed += 1
            long_win = item.long_alive and tick.bid >= item.long_profit
            short_win = item.short_alive and tick.ask <= item.short_profit
            if long_win and short_win:
                self.discarded += 1
                continue
            if long_win:
                resolved.append(ResolvedLiveLabel(item.feature_ts_ns, tick.ts_ns, item.x, 1))
                self.resolved += 1
                continue
            if short_win:
                resolved.append(ResolvedLiveLabel(item.feature_ts_ns, tick.ts_ns, item.x, 0))
                self.resolved += 1
                continue
            if item.long_alive and tick.bid <= item.long_stop:
                item.long_alive = False
            if item.short_alive and tick.ask >= item.short_stop:
                item.short_alive = False
            if not item.long_alive and not item.short_alive:
                self.discarded += 1
                continue
            if item.ticks_observed >= self.max_lookahead_ticks:
                self.discarded += 1
                continue
            keep.append(item)
        self.pending = keep
        return resolved


@dataclass
class _ServingModel:
    scaler: RobustScaler
    model: object
    generation: int
    metrics: dict

    def probability(self, vector: tuple[float, ...]) -> float:
        x = np.asarray([vector], dtype=float)
        return float(self.model.probability(self.scaler.transform(x))[0])


class AdaptiveLiveModel:
    """Rolling champion/challenger learner with live chronological validation."""

    def __init__(self, store: ScalperStore, state_dir: Path, *, recent_train_samples: int = 20000,
                 bootstrap_validation: int = 2000, live_validation: int = 120,
                 retrain_every_labels: int = 120):
        self.store = store
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.recent_train_samples = int(recent_train_samples)
        self.bootstrap_validation = int(bootstrap_validation)
        self.live_validation = int(live_validation)
        self.retrain_every_labels = int(retrain_every_labels)
        self.snapshot_path = self.state_dir / "adaptive_serving_model.json"
        self.live_labels_path = self.state_dir / "adaptive_live_labels.jsonl"
        self.historical = self._load_historical()
        self.live: list[ResolvedLiveLabel] = self._load_live_labels()
        self.new_labels_since_attempt = 0
        self.serving = self._load_snapshot() or self._bootstrap()

    def _load_historical(self) -> list[ResolvedLiveLabel]:
        limit = self.recent_train_samples + self.bootstrap_validation + 6000
        rows = list(self.store.db.execute(
            """SELECT s.feature_ts_ns,s.x_json,s.y,i.label_end_ts_ns
               FROM samples s JOIN sample_label_intervals i ON i.feature_ts_ns=s.feature_ts_ns
               ORDER BY s.id DESC LIMIT ?""",
            (int(limit),),
        ))
        rows.reverse()
        result: list[ResolvedLiveLabel] = []
        for feature_ts, x_json, y, end_ts in rows:
            x = tuple(float(v) for v in json.loads(x_json))
            if len(x) != 8 or not all(math.isfinite(v) for v in x):
                continue
            result.append(ResolvedLiveLabel(int(feature_ts), int(end_ts), x, int(y)))
        if len(result) < 12000:
            raise ValueError(f"insufficient historical labeled samples for adaptive model: {len(result)}")
        return result

    def _load_live_labels(self) -> list[ResolvedLiveLabel]:
        if not self.live_labels_path.exists():
            return []
        result: list[ResolvedLiveLabel] = []
        for line in self.live_labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                x = tuple(float(v) for v in doc["x"])
                y = int(doc["y"])
                if len(x) == 8 and y in (0, 1):
                    result.append(ResolvedLiveLabel(
                        int(doc["feature_ts_ns"]), int(doc["label_end_ts_ns"]), x, y
                    ))
            except Exception:
                continue
        return result[-10000:]

    @staticmethod
    def _fit(rows: list[ResolvedLiveLabel]):
        x = np.asarray([r.x for r in rows], dtype=float)
        y = np.asarray([r.y for r in rows], dtype=np.int8)
        scaler = RobustScaler.fit(x)
        model = fit_logit(scaler.transform(x), y, iterations=180, balanced=True)
        return scaler, model

    @staticmethod
    def _metrics(serving: _ServingModel, rows: list[ResolvedLiveLabel]) -> dict:
        x = np.asarray([r.x for r in rows], dtype=float)
        y = np.asarray([r.y for r in rows], dtype=np.int8)
        p = serving.model.probability(serving.scaler.transform(x))
        return binary_metrics(y, p)

    @staticmethod
    def _candidate_metrics(scaler: RobustScaler, model, rows: list[ResolvedLiveLabel]) -> dict:
        x = np.asarray([r.x for r in rows], dtype=float)
        y = np.asarray([r.y for r in rows], dtype=np.int8)
        return binary_metrics(y, model.probability(scaler.transform(x)))

    def _bootstrap(self) -> _ServingModel:
        validation = self.historical[-self.bootstrap_validation:]
        validation_start = validation[0].feature_ts_ns
        train = [r for r in self.historical[:-self.bootstrap_validation]
                 if r.label_end_ts_ns < validation_start]
        train = train[-self.recent_train_samples:]
        if len(train) < 10000:
            raise ValueError("insufficient purged bootstrap train history")
        scaler, model = self._fit(train)
        metrics = self._candidate_metrics(scaler, model, validation)
        serving = _ServingModel(scaler, model, 1, metrics)
        self._save_snapshot(serving)
        return serving

    def _load_snapshot(self) -> _ServingModel | None:
        if not self.snapshot_path.exists():
            return None
        try:
            doc = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            center = np.asarray(doc["center"], dtype=float)
            scale = np.asarray(doc["scale"], dtype=float)
            weights = np.asarray(doc["weights"], dtype=float)
            if center.shape != (8,) or scale.shape != (8,) or weights.shape != (8,):
                return None
            from .research_models import LogitModel
            return _ServingModel(
                RobustScaler(center, scale),
                LogitModel(weights, float(doc["bias"])),
                int(doc["generation"]),
                dict(doc.get("metrics", {})),
            )
        except Exception:
            return None

    def _save_snapshot(self, serving: _ServingModel) -> None:
        doc = {
            "generation": serving.generation,
            "center": [float(v) for v in serving.scaler.center],
            "scale": [float(v) for v in serving.scaler.scale],
            "weights": [float(v) for v in serving.model.weights],
            "bias": float(serving.model.bias),
            "metrics": serving.metrics,
        }
        self.snapshot_path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")

    @property
    def generation(self) -> int:
        return self.serving.generation

    @property
    def qualified(self) -> bool:
        bal = self.serving.metrics.get("balanced_accuracy")
        return bal is not None and float(bal) >= 0.52

    def probability(self, features: MicroFeatures) -> float:
        return self.serving.probability(features.vector())

    def add_live_labels(self, labels: list[ResolvedLiveLabel]) -> None:
        if not labels:
            return
        with self.live_labels_path.open("a", encoding="utf-8") as handle:
            for row in labels:
                handle.write(json.dumps({
                    "feature_ts_ns": row.feature_ts_ns,
                    "label_end_ts_ns": row.label_end_ts_ns,
                    "x": row.x,
                    "y": row.y,
                }, sort_keys=True) + "\n")
        self.live.extend(labels)
        self.live = self.live[-10000:]
        self.new_labels_since_attempt += len(labels)

    def maybe_retrain(self) -> dict | None:
        if self.new_labels_since_attempt < self.retrain_every_labels:
            return None
        self.new_labels_since_attempt = 0
        if len(self.live) < self.live_validation:
            return {"attempted": False, "reason": "waiting_for_live_validation", "live_labels": len(self.live)}

        validation = self.live[-self.live_validation:]
        validation_start = validation[0].feature_ts_ns
        older_live = [r for r in self.live[:-self.live_validation] if r.label_end_ts_ns < validation_start]
        historical = [r for r in self.historical if r.label_end_ts_ns < validation_start]
        train = (historical + older_live)[-self.recent_train_samples:]
        if len(train) < 10000:
            return {"attempted": False, "reason": "insufficient_purged_train", "train": len(train)}

        positives = sum(r.y for r in validation)
        minority = min(positives, len(validation) - positives)
        if minority < 20:
            return {"attempted": True, "promoted": False, "reason": "validation_class_imbalance",
                    "minority": minority}

        candidate_scaler, candidate_model = self._fit(train)
        champion_metrics = self._metrics(self.serving, validation)
        candidate_metrics = self._candidate_metrics(candidate_scaler, candidate_model, validation)
        improve = float(champion_metrics["logloss"] - candidate_metrics["logloss"])
        promoted = (
            math.isfinite(float(candidate_metrics["logloss"]))
            and improve >= 0.002
            and float(candidate_metrics["balanced_accuracy"]) >= 0.52
            and float(candidate_metrics["balanced_accuracy"]) >= float(champion_metrics["balanced_accuracy"])
        )
        result = {
            "attempted": True,
            "promoted": promoted,
            "generation_before": self.serving.generation,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "logloss_improvement": improve,
            "champion": champion_metrics,
            "challenger": candidate_metrics,
        }
        if promoted:
            self.serving = _ServingModel(
                candidate_scaler, candidate_model, self.serving.generation + 1, candidate_metrics
            )
            self._save_snapshot(self.serving)
            result["generation_after"] = self.serving.generation
        return result

    def status(self) -> dict:
        return {
            "generation": self.serving.generation,
            "qualified": self.qualified,
            "bootstrap_metrics": self.serving.metrics,
            "live_labels": len(self.live),
            "new_labels_since_attempt": self.new_labels_since_attempt,
        }
