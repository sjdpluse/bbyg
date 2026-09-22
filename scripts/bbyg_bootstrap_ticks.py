from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.learning import ChampionChallenger
from truetrade.scalper.replay import TickReplayBuilder
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution
from truetrade.scalper.trainer import SelfImprovementController
from truetrade.scalper.types import Tick


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import recent MT5 DEMO ticks into BBYG and run one replay/training cycle"
    )
    parser.add_argument("--hours", type=float, default=6.0, help="broker-history window; default 6h")
    parser.add_argument("--max-ticks", type=int, default=250_000, help="hard cap for imported ticks")
    args = parser.parse_args()
    if not 0 < args.hours <= 168:
        raise SystemExit("--hours must be in (0, 168]")
    if not 1_000 <= args.max_ticks <= 2_000_000:
        raise SystemExit("--max-ticks must be between 1,000 and 2,000,000")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ScalperStore(state_dir / "scalper.sqlite")
    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env())
    timebase = BrokerTimebase.from_env()

    try:
        broker.connect()
        raw_latest = broker.call("symbol_info_tick", broker.symbol)
        broker_end = datetime.fromtimestamp(int(raw_latest.time_msc) / 1000.0, timezone.utc)
        broker_start = broker_end - timedelta(hours=args.hours)
        rows = broker.call("copy_ticks_range", broker.symbol, broker_start, broker_end, broker.api.COPY_TICKS_ALL)
        if len(rows) > args.max_ticks:
            rows = rows[-args.max_ticks:]

        inserted = 0
        skipped_invalid = 0
        last_ns = 0
        for row in rows:
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

        replay = TickReplayBuilder().build(store)
        learner = ChampionChallenger()
        controller = SelfImprovementController(store, learner)
        cycle = controller.maybe_train()

        print(json.dumps({
            "symbol": broker.symbol,
            "server_utc_offset_seconds": timebase.utc_offset_seconds,
            "requested_hours": args.hours,
            "raw_ticks_received": len(rows),
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
                "report": None if cycle.report is None else {
                    "promoted": cycle.report.promoted,
                    "baseline_logloss": cycle.report.baseline_logloss,
                    "challenger_logloss": cycle.report.challenger_logloss,
                    "challenger_accuracy": cycle.report.challenger_accuracy,
                },
            },
        }, sort_keys=True))
    finally:
        broker.shutdown()
        store.close()


if __name__ == "__main__":
    main()
