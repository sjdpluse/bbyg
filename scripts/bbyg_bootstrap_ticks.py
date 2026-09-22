from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.learning import ChampionChallenger
from truetrade.scalper.replay import TickReplayBuilder
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution
from truetrade.scalper.trainer import SelfImprovementController
from truetrade.scalper.types import Tick


def _print(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def _status_only(store: ScalperStore) -> None:
    model = store.meta(SelfImprovementController.META_MODEL)
    events = store.events("learning_validation_consumed")
    last_learning = None if not events else events[-1]["payload"]
    _print(
        "status",
        stored_ticks=len(store.ticks()),
        samples=store.sample_count(),
        last_validation_sample_id=int(store.meta(SelfImprovementController.META_LAST_VALIDATION, 0) or 0),
        model_generation=0 if not model else int(model.get("generation", 0)),
        model_qualified=False if not model else bool(model.get("qualified", False)),
        last_learning=last_learning,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import recent MT5 DEMO ticks into BBYG and run one replay/training cycle"
    )
    parser.add_argument("--hours", type=float, default=6.0, help="broker-history window; default 6h")
    parser.add_argument("--max-ticks", type=int, default=250_000, help="hard cap for imported ticks")
    parser.add_argument("--status-only", action="store_true", help="print durable bootstrap/learning state without importing or training")
    args = parser.parse_args()
    if not 0 < args.hours <= 168:
        raise SystemExit("--hours must be in (0, 168]")
    if not 1_000 <= args.max_ticks <= 2_000_000:
        raise SystemExit("--max-ticks must be between 1,000 and 2,000,000")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ScalperStore(state_dir / "scalper.sqlite")
    if args.status_only:
        try:
            _status_only(store)
        finally:
            store.close()
        return

    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env())
    timebase = BrokerTimebase.from_env()
    started = time.perf_counter()

    try:
        _print("connecting_mt5")
        broker.connect()
        _print("connected", symbol=broker.symbol, server_utc_offset_seconds=timebase.utc_offset_seconds)

        raw_latest = broker.call("symbol_info_tick", broker.symbol)
        broker_end = datetime.fromtimestamp(int(raw_latest.time_msc) / 1000.0, timezone.utc)
        broker_start = broker_end - timedelta(hours=args.hours)
        _print("requesting_history", requested_hours=args.hours)
        rows = broker.call("copy_ticks_range", broker.symbol, broker_start, broker_end, broker.api.COPY_TICKS_ALL)
        raw_count = len(rows)
        if raw_count > args.max_ticks:
            rows = rows[-args.max_ticks:]
        _print("history_received", raw_ticks_received=raw_count, ticks_selected=len(rows))

        inserted = 0
        skipped_invalid = 0
        last_ns = 0
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            bid = float(row["bid"])
            ask = float(row["ask"])
            if bid <= 0 or ask <= 0 or ask < bid:
                skipped_invalid += 1
                continue
            last = float(row["last"])
            volume = float(row["volume_real"] if "volume_real" in row.dtype.names else row["volume"])
            ts_ns = timebase.msc_to_utc_ns(int(row["time_msc"]))
            if ts_ns <= last_ns:
                ts_ns = last_ns + 1
            last_ns = ts_ns
            inserted += int(store.append_tick(Tick(ts_ns, bid, ask, last, volume)))
            if idx % 25_000 == 0:
                _print("import_progress", processed=idx, total=total, inserted=inserted)

        _print("building_replay", stored_ticks=len(store.ticks()))
        replay = TickReplayBuilder().build(store)
        _print("replay_complete", labeled=replay.labeled, skipped=replay.skipped, samples=store.sample_count())

        learner = ChampionChallenger()
        controller = SelfImprovementController(store, learner)
        _print("training_check")
        cycle = controller.maybe_train()

        report = None if cycle.report is None else asdict(cycle.report)
        print(json.dumps({
            "stage": "complete",
            "symbol": broker.symbol,
            "server_utc_offset_seconds": timebase.utc_offset_seconds,
            "requested_hours": args.hours,
            "raw_ticks_received": raw_count,
            "ticks_selected": len(rows),
            "ticks_inserted": inserted,
            "invalid_ticks_skipped": skipped_invalid,
            "stored_ticks": len(store.ticks()),
            "replay": {
                "ticks": replay.ticks,
                "feature_rows": replay.feature_rows,
                "labeled": replay.labeled,
                "skipped": replay.skipped,
                "extra_cost_spreads": replay.extra_cost_spreads,
            },
            "samples": store.sample_count(),
            "learning": {
                "attempted": cycle.attempted,
                "reason": cycle.reason,
                "generation": learner.generation,
                "qualified": learner.qualified,
                "report": report,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, sort_keys=True), flush=True)
    finally:
        broker.shutdown()
        store.close()


if __name__ == "__main__":
    main()
