from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from truetrade.scalper.fast_replay import FastGapAwareReplayBuilder
from truetrade.scalper.labels import CostAwareLabeler, LabelSettings
from truetrade.scalper.store import ScalperStore


def _print(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BBYG multiday samples from stored ticks")
    parser.add_argument("--profit-spreads", type=float, default=1.6)
    parser.add_argument("--loss-spreads", type=float, default=1.4)
    parser.add_argument("--extra-cost-spreads", type=float, default=0.20)
    parser.add_argument("--max-lookahead-ticks", type=int, default=600)
    parser.add_argument("--max-entry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--stop-reference", choices=("exit_quote", "entry"), default="exit_quote")
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    started = time.perf_counter()

    def progress(stage: str, **fields) -> None:
        _print(stage, **fields)

    try:
        tick_count = int(store.db.execute("SELECT count(*) FROM ticks").fetchone()[0])
        if tick_count < 100_000:
            raise SystemExit("not enough stored ticks for multi-day sample build")
        settings = LabelSettings(
            profit_spreads=args.profit_spreads,
            loss_spreads=args.loss_spreads,
            extra_cost_spreads=args.extra_cost_spreads,
            max_lookahead_ticks=args.max_lookahead_ticks,
            max_entry_delay_seconds=args.max_entry_delay_seconds,
            stop_reference=args.stop_reference,
        )
        _print(
            "start",
            state_dir=str(state_dir),
            stored_ticks=tick_count,
            label_settings={
                "profit_spreads": settings.profit_spreads,
                "loss_spreads": settings.loss_spreads,
                "extra_cost_spreads": settings.extra_cost_spreads,
                "max_lookahead_ticks": settings.max_lookahead_ticks,
                "max_entry_delay_seconds": settings.max_entry_delay_seconds,
                "stop_reference": settings.stop_reference,
                "nominal_target_from_entry_spreads": settings.nominal_target_from_entry_spreads,
                "nominal_stop_from_entry_spreads": settings.nominal_stop_from_entry_spreads,
            },
        )
        builder = FastGapAwareReplayBuilder(
            labeler=CostAwareLabeler(settings),
            stride=4,
            max_gap_seconds=300.0,
            feature_batch=20_000,
            label_batch=2_500,
            write_batch=10_000,
        )
        report = builder.build(store, reset_learning=True, progress=progress)
        interval_count = int(store.db.execute("SELECT count(*) FROM sample_label_intervals").fetchone()[0])
        _print(
            "complete",
            report=asdict(report),
            stored_ticks=tick_count,
            samples=store.sample_count(),
            label_intervals=interval_count,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
