from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.runtime import DemoScalperRuntime, RuntimeSettings
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local BBYG tick scalper on an MT5 DEMO account")
    parser.add_argument("--once", action="store_true", help="process one new tick and exit")
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ScalperStore(state_dir / "scalper.sqlite")
    timebase = BrokerTimebase.from_env()
    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env(), timebase=timebase)
    runtime = DemoScalperRuntime(broker, store, settings=RuntimeSettings.from_env())

    try:
        broker.connect()
        reconciliation = runtime.startup_reconcile()
        print(json.dumps({
            "mode": "demo",
            "symbol": broker.symbol,
            "execution_enabled": runtime.settings.execution_enabled,
            "server_utc_offset_seconds": timebase.utc_offset_seconds,
            "reconciliation": reconciliation,
        }, sort_keys=True))
        if args.once:
            print(json.dumps(runtime.run_once(), sort_keys=True, default=str))
            return

        last_status = 0.0
        runtime.running = True
        while runtime.running:
            result = runtime.run_once()
            now = time.monotonic()
            if now - last_status >= 2.0:
                print(json.dumps(result, sort_keys=True, default=str), flush=True)
                last_status = now
            time.sleep(runtime.settings.poll_interval_ms / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        broker.shutdown()
        store.close()


if __name__ == "__main__":
    main()
