from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from truetrade.scalper.offline_episodes import OfflineEpisodeBuilder, OfflineEpisodeSettings
from truetrade.scalper.store import ScalperStore


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or independently replay-audit the BBYG v4 policy-independent episode dataset."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="replace and persist the v4 episode dataset")
    mode.add_argument("--audit", action="store_true", help="replay without writes and compare deterministic digest")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--horizon-ticks", type=int, default=600)
    parser.add_argument("--risk-spreads", type=float, default=2.0)
    parser.add_argument("--extra-cost-spreads", type=float, default=0.20)
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    parser.add_argument("--outcome-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--yes-replace",
        action="store_true",
        help="required with --build; confirms replacement of any existing v4 episode table contents",
    )
    args = parser.parse_args()

    if args.build and not args.yes_replace:
        raise SystemExit("--build requires --yes-replace; v4 episode construction is intentionally explicit")

    state_dir = Path(args.state_dir or os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    db_path = state_dir / "scalper.sqlite"
    if not db_path.exists():
        raise SystemExit(f"BBYG store not found: {db_path}")

    settings = OfflineEpisodeSettings(
        stride=args.stride,
        horizon_ticks=args.horizon_ticks,
        risk_spreads=args.risk_spreads,
        extra_cost_spreads=args.extra_cost_spreads,
        max_gap_ns=int(args.max_gap_seconds * 1_000_000_000),
        outcome_chunk_size=args.outcome_chunk_size,
    )
    builder = OfflineEpisodeBuilder(settings)

    with ScalperStore(db_path) as store:
        tick_count = int(store.db.execute("SELECT count(*) FROM ticks").fetchone()[0])
        bounds = store.db.execute("SELECT min(ts_ns),max(ts_ns) FROM ticks").fetchone()
        emit(
            "v4_episode_job_started",
            mode="build" if args.build else "audit",
            state_dir=str(state_dir),
            ticks=tick_count,
            min_ts_ns=None if bounds[0] is None else int(bounds[0]),
            max_ts_ns=None if bounds[1] is None else int(bounds[1]),
            settings=settings.__dict__,
            dataset_signature=builder.dataset_signature(),
            execution_authorized=False,
        )
        if tick_count == 0:
            raise SystemExit("tick store is empty")

        if args.build:
            report = builder.build(store, persist=True, replace=True)
            emit(
                "v4_episode_build_complete",
                **report.__dict__,
                memory_only=True,
                execution_authorized=False,
            )
            if report.episodes_generated == 0:
                raise SystemExit("build produced zero episodes; inspect tick cadence and warmup")
        else:
            audit = builder.parity_audit(store)
            emit("v4_episode_parity_audit", **audit, execution_authorized=False)
            if not audit["pass"]:
                raise SystemExit(2)


if __name__ == "__main__":
    main()
