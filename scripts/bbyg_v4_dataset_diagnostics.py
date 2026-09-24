from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from truetrade.scalper.dataset_diagnostics import V4DatasetDiagnostics
from truetrade.scalper.store import ScalperStore


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile BBYG v4 episode dataset and propose chronological research split.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--min-day-episodes", type=int, default=2000)
    parser.add_argument("--calibration-days", type=int, default=1)
    parser.add_argument("--validation-days", type=int, default=1)
    args = parser.parse_args()

    state_dir = Path(args.state_dir or os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    db_path = state_dir / "scalper.sqlite"
    if not db_path.exists():
        raise SystemExit(f"BBYG store not found: {db_path}")

    diagnostics = V4DatasetDiagnostics(
        min_day_episodes=args.min_day_episodes,
        minimum_calibration_days=args.calibration_days,
        minimum_validation_days=args.validation_days,
    )
    with ScalperStore(db_path) as store:
        parity = {
            "signature": store.meta("v4_episode_dataset_signature", None),
            "digest": store.meta("v4_episode_dataset_digest", None),
            "count": store.meta("v4_episode_dataset_count", None),
        }
        report = diagnostics.analyze(store)

    emit(
        "v4_dataset_diagnostics",
        execution_authorized=False,
        parity_metadata=parity,
        episode_count=report.episode_count,
        feature_dimensions=report.feature_dimensions,
        day_count=report.day_count,
        regime_counts=report.regime_counts,
        regime_fractions=report.regime_fractions,
        reward=report.reward,
        directional_edge=report.directional_edge,
        pathologies=report.pathologies,
        gates=report.gates,
        split_plan=None if report.split_plan is None else asdict(report.split_plan),
        per_day=report.per_day,
    )
    if not report.gates["all_training_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
