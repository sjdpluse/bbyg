from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
import os
import time

from .types import Intent, IntentKind, PositionState, Side, Tick

SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2


class DemoExecutionError(RuntimeError):
    pass


class ExecutionRejected(DemoExecutionError):
    pass


class ExecutionUncertain(DemoExecutionError):
    pass


@dataclass(frozen=True)
class DemoMT5Settings:
    login: int
    password: str
    server: str
    terminal_path: str
    symbol: str = "XAUUSD"
    magic: int = 731022
    deviation_points: int = 20
    max_spread_points: int = 50
    emergency_stop_points: int = 150
    risk_fraction: float = 0.0025
    commission_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if not self.login or not self.password or not self.server or not self.terminal_path:
            raise ValueError("MT5 credentials, server, and terminal path are required")
        if self.magic <= 0 or self.deviation_points < 0 or self.max_spread_points <= 0:
            raise ValueError("invalid execution settings")
        if self.emergency_stop_points <= 0:
            raise ValueError("emergency stop must be positive")
        if not 0 < self.risk_fraction <= 0.005:
            raise ValueError("demo scalper risk must be in (0, 0.5%]")
        if self.commission_per_lot < 0 or not math.isfinite(self.commission_per_lot):
            raise ValueError("invalid commission")

    @classmethod
    def from_env(cls) -> "DemoMT5Settings":
        if os.getenv("MT5_MODE", "demo").lower() != "demo":
            raise ValueError("BBYG Phase 2 execution is DEMO-only")
        return cls(
            login=int(os.environ["MT5_LOGIN"]),
            password=os.environ["MT5_PASSWORD"],
            server=os.environ["MT5_SERVER"],
            terminal_path=os.environ["MT5_TERMINAL_PATH"],
            symbol=os.getenv("BBYG_MT5_SYMBOL", "XAUUSD"),
            magic=int(os.getenv("BBYG_MT5_MAGIC", "731022")),
            deviation_points=int(os.getenv("BBYG_DEVIATION_POINTS", "20")),
            max_spread_points=int(os.getenv("BBYG_MAX_SPREAD_POINTS", "50")),
            emergency_stop_points=int(os.getenv("BBYG_EMERGENCY_STOP_POINTS", "150")),
            risk_fraction=float(os.getenv("BBYG_RISK_FRACTION", "0.0025")),
            commission_per_lot=float(os.getenv("BBYG_COMMISSION_PER_LOT", os.getenv("MT5_COMMISSION_PER_LOT", "0"))),
        )


@dataclass(frozen=True)
class ExecutionResult:
    decision_id: str
    action: str
    position_id: str | None
    filled_size: float
    expected_price: float
    fill_price: float
    order_id: int
    deal_id: int
    start_ns: int
    end_ns: int


class DemoMT5Execution:
    """Local, DEMO-only MT5 adapter with exactly-one write attempts and verification."""

    def __init__(self, settings: DemoMT5Settings, api=None):
        self.settings = settings
        if api is None:
            try:
                import MetaTrader5 as api
            except ImportError:
                raise DemoExecutionError("Install MetaTrader5 on Windows") from None
        self.api = api
        self.connected = False
        self.symbol = settings.symbol
        self._last_tick_ns = 0
        self._last_tick_signature = None

    def call(self, method: str, *args, **kwargs):
        try:
            fn = getattr(self.api, method)
            value = fn(*args, **kwargs) if kwargs else fn(*args)
        except Exception as exc:
            raise DemoExecutionError(f"MT5 {method} failed") from exc
        if value is None:
            raise DemoExecutionError(f"MT5 {method} returned no result")
        return value

    def connect(self) -> None:
        s = self.settings
        ok = self.call("initialize", s.terminal_path, login=s.login, password=s.password,
                       server=s.server, timeout=15000)
        if not ok:
            raise DemoExecutionError("MT5 initialization failed")
        self.connected = True
        self._account(trading=True)
        self.symbol = self._resolve_symbol(s.symbol)
        self._symbol_info()

    def shutdown(self) -> None:
        if self.connected:
            self.api.shutdown()
        self.connected = False

    def _account(self, *, trading: bool = False):
        if not self.connected:
            raise DemoExecutionError("MT5 not connected")
        terminal = self.call("terminal_info")
        account = self.call("account_info")
        if not terminal.connected:
            raise DemoExecutionError("MT5 terminal disconnected")
        if account.login != self.settings.login or account.server != self.settings.server:
            raise ExecutionRejected("terminal account identity changed")
        if account.trade_mode != self.api.ACCOUNT_TRADE_MODE_DEMO:
            raise ExecutionRejected("BBYG Phase 2 refuses non-DEMO accounts")
        if trading:
            if not account.trade_allowed or not account.trade_expert or not terminal.trade_allowed or terminal.tradeapi_disabled:
                raise ExecutionRejected("algorithmic trading permission denied")
            if account.margin_mode != self.api.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
                raise ExecutionRejected("hedging demo account required")
        return account

    def _resolve_symbol(self, requested: str) -> str:
        names = {x.name for x in self.call("symbols_get")}
        if requested in names:
            return requested
        matches = sorted(x for x in names if x.startswith(requested))
        if len(matches) != 1:
            raise ExecutionRejected("symbol missing or ambiguous")
        return matches[0]

    def _symbol_info(self):
        info = self.call("symbol_info", self.symbol)
        if not info.visible:
            if not self.call("symbol_select", self.symbol, True):
                raise ExecutionRejected("symbol selection failed")
            info = self.call("symbol_info", self.symbol)
        if info.trade_mode != self.api.SYMBOL_TRADE_MODE_FULL:
            raise ExecutionRejected("symbol is not fully tradeable")
        return info

    def latest_tick(self) -> Tick | None:
        self._account()
        raw = self.call("symbol_info_tick", self.symbol)
        last = float(getattr(raw, "last", 0.0) or 0.0)
        volume = float(getattr(raw, "volume_real", getattr(raw, "volume", 0.0)) or 0.0)
        signature = (int(raw.time_msc), float(raw.bid), float(raw.ask), last, volume)
        if signature == self._last_tick_signature:
            return None
        self._last_tick_signature = signature
        raw_ns = int(raw.time_msc) * 1_000_000
        ts_ns = max(raw_ns, self._last_tick_ns + 1)
        self._last_tick_ns = ts_ns
        return Tick(ts_ns, float(raw.bid), float(raw.ask), last, volume)

    @staticmethod
    def _side(position_type, api) -> Side:
        return Side.LONG if position_type == api.POSITION_TYPE_BUY else Side.SHORT

    def positions(self) -> list[PositionState]:
        self._account()
        result = []
        for p in self.call("positions_get"):
            if p.symbol != self.symbol or p.magic != self.settings.magic:
                continue
            opened_ms = int(getattr(p, "time_msc", 0) or int(getattr(p, "time", time.time())) * 1000)
            result.append(PositionState(str(p.ticket), self._side(p.type, self.api), float(p.volume),
                                        float(p.price_open), opened_ms * 1_000_000))
        return result

    def _raw_position(self, position_id: str):
        matches = self.call("positions_get", ticket=int(position_id))
        if not matches:
            return None
        if len(matches) != 1:
            raise DemoExecutionError("ambiguous position lookup")
        p = matches[0]
        if p.symbol != self.symbol or p.magic != self.settings.magic:
            raise ExecutionRejected("refusing foreign position")
        return p

    def _clean_for_entry(self) -> None:
        if any(p.symbol == self.symbol and p.magic != self.settings.magic for p in self.call("positions_get")):
            raise ExecutionRejected("foreign/manual position blocks BBYG entry")
        if any(o.symbol == self.symbol for o in self.call("orders_get")):
            raise ExecutionRejected("pending orders block BBYG entry")

    def _filling(self, info):
        if info.filling_mode & SYMBOL_FILLING_FOK:
            return self.api.ORDER_FILLING_FOK
        if info.filling_mode & SYMBOL_FILLING_IOC:
            return self.api.ORDER_FILLING_IOC
        if info.trade_exemode != self.api.SYMBOL_TRADE_EXECUTION_MARKET:
            return self.api.ORDER_FILLING_RETURN
        raise ExecutionRejected("no supported fill policy")

    def _normalized_volume(self, requested: float, info, *, allow_full_if_dust: bool = False,
                           current: float | None = None) -> float:
        step = float(info.volume_step)
        minimum = float(info.volume_min)
        maximum = float(info.volume_max)
        units = math.floor((requested + 1e-12) / step)
        volume = round(units * step, 10)
        if current is not None and allow_full_if_dust and 0 < current - volume < minimum - 1e-12:
            volume = current
        if volume < minimum - 1e-12 or volume > maximum + 1e-12:
            raise ExecutionRejected("volume outside symbol limits")
        return volume

    def _comment(self, decision_id: str) -> str:
        return "bg:" + hashlib.sha256(decision_id.encode()).hexdigest()[:20]

    def _request(self, info, side: Side, volume: float, price: float, *, decision_id: str,
                 position: int | None = None, stop: float | None = None) -> dict:
        request = {
            "action": self.api.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": float(volume),
            "type": self.api.ORDER_TYPE_BUY if side is Side.LONG else self.api.ORDER_TYPE_SELL,
            "deviation": self.settings.deviation_points, "magic": self.settings.magic,
            "comment": self._comment(decision_id), "type_time": self.api.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        if info.trade_exemode != self.api.SYMBOL_TRADE_EXECUTION_MARKET:
            request["price"] = float(price)
        if position is not None:
            request["position"] = int(position)
        if stop is not None:
            request["sl"] = float(stop)
        return request

    def _send_once(self, request: dict):
        self._account(trading=True)
        check = self.call("order_check", request)
        if type(check.retcode) is not int or check.retcode != 0:
            raise ExecutionRejected("MT5 order_check rejected request")
        self._account(trading=True)
        try:
            result = self.api.order_send(request)
        except BaseException as exc:
            raise ExecutionUncertain("MT5 order_send interrupted; do not retry") from exc
        if result is None:
            raise ExecutionUncertain("MT5 order_send returned no result; do not retry")
        retcode = getattr(result, "retcode", None)
        order = getattr(result, "order", None)
        deal = getattr(result, "deal", None)
        if type(retcode) is not int or type(order) is not int or type(deal) is not int:
            raise ExecutionUncertain("malformed MT5 send response; do not retry")
        if retcode not in {self.api.TRADE_RETCODE_DONE, self.api.TRADE_RETCODE_DONE_PARTIAL}:
            if not order and not deal:
                raise ExecutionRejected(f"MT5 order rejected retcode={retcode}")
            raise ExecutionUncertain(f"uncertain MT5 retcode={retcode}; do not retry")
        return result

    def _verify_deal(self, result, expected_side: Side):
        if not result.order or not result.deal:
            raise ExecutionUncertain("missing order/deal identifiers")
        orders = [o for o in self.call("history_orders_get", ticket=int(result.order)) if o.ticket == result.order]
        if len(orders) != 1:
            raise ExecutionUncertain("order history not uniquely verified")
        order = orders[0]
        position_identifier = int(order.position_id)
        deals = [d for d in self.call("history_deals_get", position=position_identifier) if d.ticket == result.deal]
        if len(deals) != 1:
            raise ExecutionUncertain("deal history not uniquely verified")
        deal = deals[0]
        expected_type = self.api.DEAL_TYPE_BUY if expected_side is Side.LONG else self.api.DEAL_TYPE_SELL
        if (deal.order != result.order or deal.position_id != position_identifier or deal.symbol != self.symbol or
                deal.magic != self.settings.magic or deal.type != expected_type):
            raise ExecutionUncertain("verified deal ownership mismatch")
        return deal, position_identifier

    def _risk_checked_stop(self, side: Side, volume: float, entry: float, info) -> float:
        point = float(info.point)
        distance = max(self.settings.emergency_stop_points, int(info.trade_stops_level) + 1) * point
        stop = round(entry - distance if side is Side.LONG else entry + distance, int(info.digits))
        kind = self.api.ORDER_TYPE_BUY if side is Side.LONG else self.api.ORDER_TYPE_SELL
        pnl = self.call("order_calc_profit", kind, self.symbol, volume, entry, stop)
        risk = max(0.0, -float(pnl)) + volume * self.settings.commission_per_lot
        equity = float(self._account().equity)
        if risk <= 0 or risk > equity * self.settings.risk_fraction + 1e-9:
            raise ExecutionRejected("requested size exceeds demo emergency-stop risk budget")
        return stop

    def _fresh_tick_for_write(self) -> Tick:
        tick = self.latest_tick()
        if tick is not None:
            return tick
        raw = self.call("symbol_info_tick", self.symbol)
        return Tick(max(int(raw.time_msc) * 1_000_000, self._last_tick_ns + 1), float(raw.bid), float(raw.ask),
                    float(getattr(raw, "last", 0.0) or 0.0),
                    float(getattr(raw, "volume_real", getattr(raw, "volume", 0.0)) or 0.0))

    def open(self, side: Side, size: float, decision_id: str) -> ExecutionResult:
        self._account(trading=True)
        self._clean_for_entry()
        info = self._symbol_info()
        tick = self._fresh_tick_for_write()
        if tick.spread / float(info.point) > self.settings.max_spread_points:
            raise ExecutionRejected("spread limit exceeded")
        volume = self._normalized_volume(size, info)
        expected = tick.ask if side is Side.LONG else tick.bid
        stop = self._risk_checked_stop(side, volume, expected, info)
        request = self._request(info, side, volume, expected, decision_id=decision_id, stop=stop)
        start = time.perf_counter_ns()
        result = self._send_once(request)
        end = time.perf_counter_ns()
        deal, identifier = self._verify_deal(result, side)
        matches = [p for p in self.call("positions_get")
                   if p.symbol == self.symbol and p.magic == self.settings.magic and p.identifier == identifier]
        if len(matches) != 1:
            raise ExecutionUncertain("filled position not uniquely observable")
        p = matches[0]
        if abs(float(p.volume) - volume) > max(float(info.volume_step) / 2, 1e-12):
            raise ExecutionUncertain("unexpected fill volume")
        if float(getattr(p, "sl", 0.0)) <= 0:
            raise ExecutionUncertain("emergency stop not observable")
        return ExecutionResult(decision_id, "OPEN", str(p.ticket), float(deal.volume), expected,
                               float(deal.price), int(result.order), int(result.deal), start, end)

    def close(self, position_id: str, fraction: float, decision_id: str) -> ExecutionResult:
        self._account(trading=True)
        p = self._raw_position(position_id)
        if p is None:
            raise ExecutionRejected("position already absent")
        info = self._symbol_info()
        current = float(p.volume)
        volume = self._normalized_volume(current * fraction, info, allow_full_if_dust=True, current=current)
        closing_side = Side.SHORT if self._side(p.type, self.api) is Side.LONG else Side.LONG
        tick = self._fresh_tick_for_write()
        expected = tick.ask if closing_side is Side.LONG else tick.bid
        request = self._request(info, closing_side, volume, expected, decision_id=decision_id, position=int(p.ticket))
        start = time.perf_counter_ns()
        result = self._send_once(request)
        end = time.perf_counter_ns()
        deal, _identifier = self._verify_deal(result, closing_side)
        if int(deal.position_id) != int(p.identifier):
            raise ExecutionUncertain("exit deal position mismatch")
        after = self._raw_position(position_id)
        if volume >= current - 1e-12:
            if after is not None:
                raise ExecutionUncertain("full close not confirmed")
            remaining_id = None
        else:
            expected_remaining = current - volume
            if after is None or abs(float(after.volume) - expected_remaining) > max(float(info.volume_step) / 2, 1e-12):
                raise ExecutionUncertain("partial reduction not confirmed")
            remaining_id = position_id
        action = "CLOSE" if fraction >= 1.0 - 1e-12 else "REDUCE"
        return ExecutionResult(decision_id, action, remaining_id, float(deal.volume), expected,
                               float(deal.price), int(result.order), int(result.deal), start, end)

    def execute(self, intent: Intent, decision_id: str) -> ExecutionResult | None:
        if intent.kind is IntentKind.HOLD:
            return None
        if intent.kind in {IntentKind.OPEN, IntentKind.ADD}:
            if intent.side is None or intent.size is None:
                raise ExecutionRejected("entry intent missing side/size")
            return self.open(intent.side, intent.size, decision_id)
        if intent.kind in {IntentKind.REDUCE, IntentKind.CLOSE}:
            if intent.position_id is None:
                raise ExecutionRejected("exit intent missing position id")
            return self.close(intent.position_id, 1.0 if intent.kind is IntentKind.CLOSE else intent.fraction, decision_id)
        raise ExecutionRejected("unsupported intent")

    def find_decision(self, decision_id: str, *, hours: int = 24):
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        deals = self.call("history_deals_get", start, end)
        matches = [d for d in deals if d.magic == self.settings.magic and d.comment == self._comment(decision_id)]
        if len(matches) > 1:
            raise DemoExecutionError("ambiguous decision history")
        return None if not matches else matches[0]
