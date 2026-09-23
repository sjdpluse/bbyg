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


def _risk_probe(broker, settings: DemoMT5Settings, side: Side, requested_size: float) -> dict:
    account = broker._account(trading=True)
    info = broker._symbol_info()
    raw = broker.call("symbol_info_tick", broker.symbol)
    point = float(info.point)
    tick_size = float(info.trade_tick_size)
    minimum = float(info.volume_min)
    maximum = float(info.volume_max)
    step = float(info.volume_step)
    requested = max(minimum, requested_size)
    units = int((requested + 1e-12) // step)
    volume = round(units * step, 10)
    if volume < minimum:
        volume = minimum
    if volume > maximum:
        raise SystemExit("Requested smoke size exceeds symbol maximum volume")

    entry = float(raw.ask if side is Side.LONG else raw.bid)
    distance_points = max(settings.emergency_stop_points, int(info.trade_stops_level) + 1)
    distance = distance_points * point
    stop = round(entry - distance if side is Side.LONG else entry + distance, int(info.digits))
    order_type = broker.api.ORDER_TYPE_BUY if side is Side.LONG else broker.api.ORDER_TYPE_SELL
    pnl = broker.call("order_calc_profit", order_type, broker.symbol, volume, entry, stop)
    estimated_risk = max(0.0, -float(pnl)) + volume * settings.commission_per_lot
    equity = float(account.equity)
    allowed_risk = equity * settings.risk_fraction
    required_fraction = 0.0 if equity <= 0 else estimated_risk / equity
    return {
        "equity": equity,
        "configured_risk_fraction": settings.risk_fraction,
        "allowed_risk_amount": allowed_risk,
        "estimated_emergency_stop_risk": estimated_risk,
        "required_risk_fraction": required_fraction,
        "volume_min": minimum,
        "volume_step": step,
        "normalized_volume": volume,
        "point": point,
        "trade_tick_size": tick_size,
        "trade_stops_level": int(info.trade_stops_level),
        "emergency_stop_points": settings.emergency_stop_points,
        "entry": entry,
        "emergency_stop": stop,
        "within_budget": estimated_risk <= allowed_risk + 1e-9,
        "max_supported_demo_risk_fraction": 0.005,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open and close one tiny REAL MT5 DEMO trade to smoke-test BBYG execution plumbing"
    )
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--probe-only", action="store_true", help="print exact demo risk budget and do not place an order")
    args = parser.parse_args()

    if os.getenv("MT5_MODE", "demo").lower() != "demo":
        raise SystemExit("Refusing to run: MT5_MODE must be demo")
    if not args.probe_only and os.getenv("BBYG_SMOKE_TRADE_CONFIRM", "") != "I_UNDERSTAND_DEMO_ONLY":
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

        existing = broker.positions()
        if existing:
            raise SystemExit(f"Refusing smoke trade: found {len(existing)} existing BBYG position(s)")

        risk = _risk_probe(broker, settings, side, args.size)
        _print("risk_probe", **risk)
        if args.probe_only:
            _print("complete", success=True, probe_only=True)
            return
        if not risk["within_budget"]:
            _print(
                "rejected",
                success=False,
                error="requested size exceeds demo emergency-stop risk budget",
                guidance=(
                    "Do not bypass the guard blindly. If required_risk_fraction is <= 0.005, "
                    "the smoke test can be retried with BBYG_RISK_FRACTION set deliberately "
                    "to at least that value; otherwise use a larger DEMO equity balance."
                ),
            )
            raise SystemExit(2)

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
