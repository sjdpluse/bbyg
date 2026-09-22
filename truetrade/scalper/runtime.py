from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import time

from .engine import ScalperCore
from .execution import DemoMT5Execution, ExecutionRejected, ExecutionResult, ExecutionUncertain
from .replay import TickReplayBuilder
from .store import ScalperStore
from .telemetry import ExecutionTelemetry
from .trainer import SelfImprovementController
from .types import Intent, IntentKind, PositionState, Side, Tick


@dataclass(frozen=True)
class RuntimeSettings:
    poll_interval_ms: int = 10
    replay_every_ticks: int = 2000
    execution_enabled: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.poll_interval_ms <= 1000:
            raise ValueError("poll_interval_ms out of range")
        if self.replay_every_ticks < 100:
            raise ValueError("replay cadence too small")

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            poll_interval_ms=int(os.getenv("BBYG_POLL_INTERVAL_MS", "10")),
            replay_every_ticks=int(os.getenv("BBYG_REPLAY_EVERY_TICKS", "2000")),
            execution_enabled=os.getenv("BBYG_DEMO_EXECUTION", "false").lower() == "true",
        )


class DemoScalperRuntime:
    """Single-machine local scalper loop with durable recovery and guarded learning."""

    def __init__(self, broker: DemoMT5Execution, store: ScalperStore, *,
                 core: ScalperCore | None = None, settings: RuntimeSettings | None = None):
        self.broker = broker
        self.store = store
        self.settings = settings or RuntimeSettings()
        self.core = core or ScalperCore()
        self.replay = TickReplayBuilder()
        self.learning = SelfImprovementController(store, self.core.model)
        self.telemetry = ExecutionTelemetry()
        self.tick_count = 0
        self.running = False

    @staticmethod
    def _decision_id(tick: Tick, index: int, intent: Intent) -> str:
        raw = f"{tick.ts_ns}:{index}:{intent.kind.value}:{intent.position_id or ''}:{intent.side.value if intent.side else ''}"
        return "sc_" + hashlib.sha256(raw.encode()).hexdigest()[:40]

    @staticmethod
    def _merge_tracking(fresh: list[PositionState], prior: list[PositionState]) -> list[PositionState]:
        old = {p.position_id: p for p in prior}
        for p in fresh:
            q = old.get(p.position_id)
            if q is not None:
                p.peak_exit_price = q.peak_exit_price
                p.trough_exit_price = q.trough_exit_price
                p.reductions = q.reductions
        return fresh

    def _positions(self) -> list[PositionState]:
        fresh = self.broker.positions()
        merged = self._merge_tracking(fresh, self.store.load_positions())
        self.store.replace_positions(merged)
        return merged

    def startup_reconcile(self) -> dict:
        unresolved = self.store.unsettled_execution_intents()
        for record in unresolved:
            state = record["state"]
            decision_id = record["decision_id"]
            if state == "created":
                self.store.transition_execution_intent(decision_id, "rejected", detail="crash_before_submit")
                continue
            deal = self.broker.find_decision(decision_id)
            if deal is not None:
                self.store.transition_execution_intent(
                    decision_id, "confirmed", broker_position_id=str(deal.position_id),
                    detail="reconciled_from_mt5_deal_history",
                )
        remaining = self.store.unsettled_execution_intents()
        self.store.set_meta("scalper_execution_halted", bool(remaining))
        return {"unsettled": len(remaining), "halted": bool(remaining)}

    def _execution_side(self, intent: Intent, positions_before: list[PositionState]) -> Side:
        if intent.kind in {IntentKind.OPEN, IntentKind.ADD}:
            if intent.side is None:
                raise ValueError("entry side missing")
            return intent.side
        p = next((x for x in positions_before if x.position_id == intent.position_id), None)
        if p is None:
            raise ValueError("exit position missing from snapshot")
        return Side.SHORT if p.side is Side.LONG else Side.LONG

    def _execute(self, tick: Tick, index: int, intent: Intent,
                 positions_before: list[PositionState]) -> ExecutionResult | None:
        if intent.kind is IntentKind.HOLD:
            return None
        decision_id = self._decision_id(tick, index, intent)
        created = self.store.create_execution_intent(
            decision_id, tick.ts_ns, kind=intent.kind.value, position_id=intent.position_id,
            side=intent.side.value if intent.side else None, size=intent.size,
            fraction=intent.fraction, detail=intent.reason,
        )
        if not created:
            record = self.store.execution_intent(decision_id)
            if record and record["state"] == "confirmed":
                return None
            raise ExecutionUncertain("duplicate unresolved decision id")

        self.store.transition_execution_intent(decision_id, "submitted", detail="write_attempt_started")
        self.core.risk.note_order()
        execution_side = self._execution_side(intent, positions_before)
        try:
            result = self.broker.execute(intent, decision_id)
        except ExecutionRejected as exc:
            self.store.transition_execution_intent(decision_id, "rejected", detail=str(exc))
            self.store.append_event(tick.ts_ns, "execution_rejected", {"decision_id": decision_id, "reason": str(exc)})
            return None
        except ExecutionUncertain as exc:
            self.store.transition_execution_intent(decision_id, "unknown", detail=str(exc))
            self.store.set_meta("scalper_execution_halted", True)
            self.store.append_event(tick.ts_ns, "execution_uncertain", {"decision_id": decision_id, "reason": str(exc)})
            raise

        if result is None:
            raise ExecutionUncertain("non-HOLD execution returned no result")
        self.store.transition_execution_intent(
            decision_id, "confirmed", broker_position_id=result.position_id, detail="mt5_fill_verified",
        )
        self.telemetry.record(
            start_ns=result.start_ns, end_ns=result.end_ns,
            expected_price=result.expected_price, fill_price=result.fill_price,
            spread=max(tick.spread, 1e-12), side=execution_side,
            size=max(result.filled_size, 1e-12), success=True,
        )
        self.store.append_event(
            tick.ts_ns, "execution_confirmed",
            {"decision_id": decision_id, "action": result.action, "position_id": result.position_id,
             "filled_size": result.filled_size, "order_id": result.order_id, "deal_id": result.deal_id},
        )
        if intent.kind in {IntentKind.OPEN, IntentKind.ADD} and intent.side is not None:
            self.core.note_entry(intent.side, tick)
        return result

    def _maybe_learn(self) -> dict | None:
        if self.tick_count == 0 or self.tick_count % self.settings.replay_every_ticks:
            return None
        replay = self.replay.build(self.store)
        cycle = self.learning.maybe_train()
        result = {"replay": asdict(replay), "learning": {"attempted": cycle.attempted, "reason": cycle.reason}}
        if cycle.report is not None:
            result["learning"]["report"] = asdict(cycle.report)
        self.store.append_event(time.time_ns(), "learning_cycle", result)
        return result

    def run_once(self) -> dict:
        tick = self.broker.latest_tick()
        if tick is None:
            return {"result": "no_new_tick"}
        if not self.store.append_tick(tick):
            return {"result": "duplicate_tick"}
        self.tick_count += 1
        positions = self._positions()
        intents = self.core.on_tick(tick, positions)
        halted = bool(self.store.meta("scalper_execution_halted", False))

        executed = 0
        if self.settings.execution_enabled and not halted:
            for idx, intent in enumerate(intents):
                if intent.kind is IntentKind.HOLD:
                    continue
                self._execute(tick, idx, intent, positions)
                executed += 1
                positions = self._positions()
        else:
            for intent in intents:
                if intent.kind is not IntentKind.HOLD:
                    self.store.append_event(
                        tick.ts_ns, "intent_observed_no_write",
                        {"kind": intent.kind.value, "reason": intent.reason, "confidence": intent.confidence},
                    )

        learning = self._maybe_learn()
        return {
            "result": "processed", "tick_ns": tick.ts_ns, "positions": len(positions),
            "intents": len(intents), "executed": executed,
            "execution_enabled": self.settings.execution_enabled,
            "halted": bool(self.store.meta("scalper_execution_halted", False)),
            "model_generation": self.core.model.generation,
            "model_qualified": self.core.model.qualified,
            "learning": learning, "telemetry": self.telemetry.summary(),
        }

    def run_forever(self) -> None:
        self.running = True
        while self.running:
            try:
                self.run_once()
            except ExecutionUncertain:
                pass
            time.sleep(self.settings.poll_interval_ms / 1000.0)

    def stop(self) -> None:
        self.running = False
