from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from truetrade.scalper.fast_replay import FastGapAwareReplayBuilder
from truetrade.scalper.store import ScalperStore


def _print(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def main() -> None:
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    started = time.perf_counter()

    def progress(stage: str, **fields) -> None:
        _print(stage, **fields)

    try:
        tick_count = int(store.db.execute("SELECT count(*) FROM ticks").fetchone()[0])
        if tick_count < 100_000:
            raise SystemExit("not enough stored ticks for multi-day sample build")
        _print("start", state_dir=str(state_dir), stored_ticks=tick_count)
        builder = FastGapAwareReplayBuilder(
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
