from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .store import ScalperStore


@dataclass(frozen=True)
class ForwardQualificationSettings:
    min_trades: int = 100
    min_days: float = 30.0
    min_profit_factor: float = 1.20
    max_drawdown_fraction: float = 0.10
    max_latency_p95_ms: float = 500.0
    max_slippage_p95_spreads: float = 1.0

    def __post_init__(self) -> None:
        if self.min_trades < 30 or self.min_days <= 0:
            raise ValueError("forward evidence too small")
        if self.min_profit_factor <= 1 or not 0 < self.max_drawdown_fraction < 1:
            raise ValueError("invalid forward thresholds")


@dataclass(frozen=True)
class ForwardQualificationReport:
    generation: int
    eligible: bool
    trades: int
    days: float
    net_pnl: float
    profit_factor: float | None
    max_drawdown_fraction: float
    expectancy_r: float | None
    expectancy_lower95_r: float | None
    reasons: tuple[str, ...]


class ForwardDemoQualifier:
    """Forward-demo gate. It records evidence only and never enables live trading."""

    META_PREFIX = "scalper_forward_qualification_generation_"

    def __init__(self, settings: ForwardQualificationSettings | None = None):
        self.settings = settings or ForwardQualificationSettings()

    @staticmethod
    def _lower95_r(values: np.ndarray) -> float | None:
        if len(values) < 30 or not np.isfinite(values).all():
            return None
        rng = np.random.default_rng(731022)
        block = min(5, len(values))
        means = []
        for _ in range(1000):
            starts = rng.integers(0, len(values) - block + 1,
                                  size=int(np.ceil(len(values) / block)))
            sample = np.concatenate([values[s:s + block] for s in starts])[:len(values)]
            means.append(float(sample.mean()))
        return float(np.quantile(means, 0.025))

    def evaluate(self, rows: list[dict], generation: int,
                 execution_summary: dict | None = None) -> ForwardQualificationReport:
        cfg = self.settings
        rows = [r for r in rows if int(r["model_generation"]) == int(generation)]
        rows.sort(key=lambda r: int(r["closed_ns"]))
        reasons: list[str] = []
        trades = len(rows)
        if len({r["decision_id"] for r in rows}) != trades or len({r["position_id"] for r in rows}) != trades:
            reasons.append("duplicate_forward_trade_identity")

        days = 0.0
        net = 0.0
        pf = None
        drawdown = 0.0
        expectancy = None
        lower = None
        if rows:
            days = (int(rows[-1]["closed_ns"]) - int(rows[0]["opened_ns"])) / 86_400_000_000_000
            pnl = np.asarray([float(r["net_pnl"]) for r in rows], dtype=float)
            risk = np.asarray([float(r["risk_amount"]) for r in rows], dtype=float)
            equity0 = float(rows[0]["entry_equity"])
            if not np.isfinite(pnl).all() or not np.isfinite(risk).all() or equity0 <= 0 or np.any(risk <= 0):
                reasons.append("invalid_forward_evidence")
            else:
                net = float(pnl.sum())
                wins = float(pnl[pnl > 0].sum())
                losses = float(-pnl[pnl < 0].sum())
                pf = None if losses <= 0 else wins / losses
                curve = equity0 + np.cumsum(pnl)
                peaks = np.maximum.accumulate(np.concatenate(([equity0], curve)))[:-1]
                drawdown = float(np.max(np.maximum(0.0, peaks - curve) / np.maximum(peaks, 1e-12)))
                r = pnl / risk
                expectancy = float(r.mean())
                lower = self._lower95_r(r)

        if trades < cfg.min_trades:
            reasons.append("insufficient_forward_trades")
        if days < cfg.min_days:
            reasons.append("insufficient_forward_days")
        if net <= 0:
            reasons.append("nonpositive_forward_net_pnl")
        if pf is None or pf < cfg.min_profit_factor:
            reasons.append("forward_profit_factor_below_threshold")
        if drawdown > cfg.max_drawdown_fraction:
            reasons.append("forward_drawdown_above_threshold")
        if lower is None or lower <= 0:
            reasons.append("forward_expectancy_uncertain_or_negative")

        execution_summary = execution_summary or {}
        latency = execution_summary.get("latency_p95_ms")
        slippage = execution_summary.get("slippage_p95_spreads")
        if latency is None or float(latency) > cfg.max_latency_p95_ms:
            reasons.append("forward_execution_latency_unqualified")
        if slippage is None or float(slippage) > cfg.max_slippage_p95_spreads:
            reasons.append("forward_execution_slippage_unqualified")

        return ForwardQualificationReport(
            generation=int(generation), eligible=not reasons, trades=trades, days=days,
            net_pnl=net, profit_factor=pf, max_drawdown_fraction=drawdown,
            expectancy_r=expectancy, expectancy_lower95_r=lower, reasons=tuple(reasons),
        )

    def evaluate_store(self, store: ScalperStore, generation: int,
                       execution_summary: dict | None = None) -> ForwardQualificationReport:
        report = self.evaluate(store.trade_outcomes(generation), generation, execution_summary)
        store.set_meta(self.META_PREFIX + str(generation), asdict(report))
        return report

    @staticmethod
    def performance_multiplier(report: ForwardQualificationReport | None) -> float:
        if report is None or report.trades < 20:
            return 0.50
        if report.net_pnl <= 0 or report.max_drawdown_fraction > 0.08:
            return 0.25
        if report.eligible:
            return 1.00
        if report.profit_factor is not None and report.profit_factor >= 1.10:
            return 0.70
        return 0.45
