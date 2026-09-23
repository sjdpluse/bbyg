from __future__ import annotations

import argparse
import json
import os
import time
import uuid

from truetrade.scalper.execution import DemoMT5Settings, ExecutionRejected, ExecutionUncertain
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution
from truetrade.scalper.types import Side


def _print(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open and close one tiny REAL MT5 DEMO trade to smoke-test BBYG execution plumbing"
    )
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if os.getenv("MT5_MODE", "demo").lower() != "demo":
        raise SystemExit("Refusing to run: MT5_MODE must be demo")
    if os.getenv("BBYG_SMOKE_TRADE_CONFIRM", "") != "I_UNDERSTAND_DEMO_ONLY":
        raise SystemExit(
            "Refusing to trade. Set BBYG_SMOKE_TRADE_CONFIRM=I_UNDERSTAND_DEMO_ONLY "
            "in this PowerShell session after confirming the terminal is logged into the intended DEMO account."
        )
    if not 0 < args.size <= 0.01:
        raise SystemExit("Smoke test size must be in (0, 0.01]")
    if not 0.5 <= args.hold_seconds <= 15:
        raise SystemExit("--hold-seconds must be between 0.5 and 15")

    settings = DemoMT5Settings.from_env()
    timebase = BrokerTimebase.from_env()
    broker = TimeNormalizedDemoMT5Execution(settings, timebase=timebase)
    side = Side.LONG if args.side == "long" else Side.SHORT
    open_id = "smoke_open_" + uuid.uuid4().hex
    close_id = "smoke_close_" + uuid.uuid4().hex
    opened = None

    try:
        broker.connect()
        account = broker._account(trading=True)
        _print(
            "connected",
            login=int(account.login),
            server=str(account.server),
            trade_mode=int(account.trade_mode),
            symbol=broker.symbol,
            requested_side=args.side,
            requested_size=args.size,
            demo_only=True,
        )

        # Do not run the smoke trade on top of an existing BBYG position.
        existing = broker.positions()
        if existing:
            raise SystemExit(f"Refusing smoke trade: found {len(existing)} existing BBYG position(s)")

        opened = broker.open(side, args.size, open_id)
        _print(
            "opened",
            action=opened.action,
            position_id=opened.position_id,
            filled_size=opened.filled_size,
            expected_price=opened.expected_price,
            fill_price=opened.fill_price,
            order_id=opened.order_id,
            deal_id=opened.deal_id,
            position_identifier=opened.position_identifier,
            emergency_risk_amount=opened.risk_amount,
        )

        time.sleep(args.hold_seconds)

        if opened.position_id is None:
            raise RuntimeError("opened trade has no position id")
        closed = broker.close(opened.position_id, 1.0, close_id)
        _print(
            "closed",
            action=closed.action,
            filled_size=closed.filled_size,
            expected_price=closed.expected_price,
            fill_price=closed.fill_price,
            order_id=closed.order_id,
            deal_id=closed.deal_id,
            position_identifier=closed.position_identifier,
        )

        outcome = None
        if opened.position_identifier is not None:
            for _ in range(10):
                outcome = broker.closed_outcome(opened.position_identifier)
                if outcome is not None:
                    break
                time.sleep(0.25)
        _print("complete", success=True, outcome=outcome)

    except ExecutionRejected as exc:
        _print("rejected", success=False, error=str(exc))
        raise SystemExit(2) from None
    except ExecutionUncertain as exc:
        _print(
            "uncertain",
            success=False,
            error=str(exc),
            warning="Do not retry blindly. Inspect MT5 terminal/positions first.",
            opened_position_id=None if opened is None else opened.position_id,
        )
        raise SystemExit(3) from None
    finally:
        broker.shutdown()


if __name__ == "__main__":
    main()
