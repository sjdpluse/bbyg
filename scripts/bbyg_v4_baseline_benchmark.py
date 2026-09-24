from __future__ import annotations

import json
import os
from pathlib import Path

from truetrade.scalper.store import ScalperStore
from truetrade.scalper.v4_baselines import BaselineSettings, run_v4_baselines


def main() -> None:
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    db_path = state_dir / "scalper.sqlite"
    if not db_path.exists():
        raise SystemExit(f"BBYG store not found: {db_path}")
    with ScalperStore(db_path) as store:
        result = run_v4_baselines(store, BaselineSettings())
    print(json.dumps({"stage": "v4_baseline_benchmark", **result}, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
