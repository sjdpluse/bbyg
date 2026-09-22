from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import time

from .engine import ScalperCore
from .execution import DemoMT5Execution, ExecutionRejected, ExecutionResult, ExecutionUncertain
from .forward import ForwardDemoQualifier, ForwardQualificationReport
from .quality import ExecutionQualityController
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
    """Local tick runtime with durable recovery, adaptive execution, and forward-demo gates."""

    def __init__(self, broker: DemoMT5Execution, store: ScalperStore, *,
                 core: ScalperCore | None = None, settings: RuntimeSettings | None = None):
        self.broker = broker
        self.store = store
        self.settings = settings or RuntimeSettings()
        self.core = core or ScalperCore()
        self.replay = TickReplayBuilder()
        self.learning = SelfImprovementController(store, self.core.model)
        self.telemetry = ExecutionTelemetry()
        self.quality = ExecutionQualityController(store)
        self.forward = ForwardDemoQualifier()
        self.forward_report: ForwardQualificationReport | None = None
        self.tick_count = 0
        self.running = False
        self._restore_telemetry()
        self._restore_forward_report()

    @staticmethod
    def _decision_id(tick: Tick, index: int, intent: Intent) -> str:
        raw = (f"{tick.ts_ns}:{index}:{intent.kind.value}:{intent.position_id or ''}:"
               f"{intent.side.value if intent.side else ''}:{intent.stop or ''}:{intent.size or ''}")
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
                if p.broker_stop is None:
                    p.broker_stop = q.broker_stop
        return fresh

    def _restore_telemetry(self) -> None:
        events = self.store.events("execution_telemetry")
        for event in events[-self.telemetry.max_samples:]:
            try:
                self.telemetry.observe(self.telemetry.deserialize(event["payload"]))
            except (KeyError, TypeError, ValueError):
                continue

    def _restore_forward_report(self) -> None:
        generation = self.core.model.generation
        if generation <= 0:
            return
        saved = self.store.meta(self.forward.META_PREFIX + str(generation))
        if saved:
            try:
                self.forward_report = ForwardQualificationReport(
                    generation=int(saved["generation"]), eligible=bool(saved["eligible"]),
                    trades=int(saved["trades"]), days=float(saved["days"]),
                    net_pnl=float(saved["net_pnl"]),
                    profit_factor=None if saved["profit_factor"] is None else float(saved["profit_factor"]),
                    max_drawdown_fraction=float(saved["max_drawdown_fraction"]),
                    expectancy_r=None if saved["expectancy_r"] is None else float(saved["expectancy_r"]),
                    expectancy_lower95_r=None if saved["expectancy_lower95_r"] is None else float(saved["expectancy_lower95_r"]),
                    reasons=tuple(saved["reasons"]),
                )
            except (KeyError, TypeError, ValueError):
                self.forward_report = None

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

            if record["kind"] == IntentKind.PROTECT.value:
                position_id = record["position_id"]
                stop = record["stop"]
                if (position_id is not None and stop is not None
                        and self.broker.protection_matches(str(position_id), float(stop))):
                    self.store.transition_execution_intent(
                        decision_id, "confirmed", broker_position_id=str(position_id),
                        detail="reconciled_from_protection_readback",
                    )
                continue

            deal = self.broker.find_decision(decision_id)
            if deal is not None:
                identifier = int(deal.position_id)
                ticket = self.broker.position_ticket_by_identifier(identifier)
                self.store.transition_execution_intent(
                    decision_id, "confirmed",
                    broker_position_id=ticket or record["broker_position_id"] or str(identifier),
                    broker_identifier=identifier, detail="reconciled_from_mt5_deal_history",
                )

        remaining = self.store.unsettled_execution_intents()
        self.store.set_meta("scalper_execution_halted", bool(remaining))
        return {"unsettled": len(remaining), "halted": bool(remaining)}

    def _execution_side(self, intent: Intent, positions_before: list[PositionState]) -> Side | None:
        if intent.kind in {IntentKind.OPEN, IntentKind.ADD}:
            if intent.side is None:
                raise ValueError("entry side missing")
            return intent.side
        if intent.kind is IntentKind.PROTECT:
            return None
        p = next((x for x in positions_before if x.position_id == intent.position_id), None)
        if p is None:
            raise ValueError("exit position missing from snapshot")
        return Side.SHORT if p.side is Side.LONG else Side.LONG

    def _execute(self, tick: Tick, index: int, intent: Intent,
                 positions_before: list[PositionState]) -> ExecutionResult | None:
        if intent.kind is IntentKind.HOLD:
            return None
        decision_id = self._decision_id(tick, index, intent)
        entry_equity = self.broker.account_equity() if intent.kind in {IntentKind.OPEN, IntentKind.ADD} else None
        created = self.store.create_execution_intent(
            decision_id, tick.ts_ns, kind=intent.kind.value, position_id=intent.position_id,
            side=intent.side.value if intent.side else None, size=intent.size,
            fraction=intent.fraction, stop=intent.stop, detail=intent.reason,
            model_generation=self.core.model.generation, entry_equity=entry_equity,
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
            self.quality.observe_failure(uncertain=False)
            self.store.append_event(tick.ts_ns, "execution_rejected",
                                    {"decision_id": decision_id, "reason": str(exc), "kind": intent.kind.value})
            return None
        except ExecutionUncertain as exc:
            self.store.transition_execution_intent(decision_id, "unknown", detail=str(exc))
            self.quality.observe_failure(uncertain=True)
            self.store.set_meta("scalper_execution_halted", True)
            self.store.append_event(tick.ts_ns, "execution_uncertain",
                                    {"decision_id": decision_id, "reason": str(exc), "kind": intent.kind.value})
            raise

        if result is None:
            raise ExecutionUncertain("non-HOLD execution returned no result")
        self.store.transition_execution_intent(
            decision_id, "confirmed", broker_position_id=result.position_id,
            broker_identifier=result.position_identifier, risk_amount=result.risk_amount,
            detail="mt5_result_verified",
        )

        if execution_side is not None and result.filled_size > 0:
            obs = self.telemetry.record(
                start_ns=result.start_ns, end_ns=result.end_ns,
                expected_price=result.expected_price, fill_price=result.fill_price,
                spread=max(tick.spread, 1e-12), side=execution_side,
                size=max(result.filled_size, 1e-12), success=True,
            )
            self.quality.observe(obs)
            self.store.append_event(tick.ts_ns, "execution_telemetry", self.telemetry.serialize(obs))

        self.store.append_event(
            tick.ts_ns, "execution_confirmed",
            {"decision_id": decision_id, "action": result.action,
             "position_id": result.position_id, "position_identifier": result.position_identifier,
             "filled_size": result.filled_size, "order_id": result.order_id,
             "deal_id": result.deal_id, "risk_amount": result.risk_amount},
        )
        if intent.kind in {IntentKind.OPEN, IntentKind.ADD} and intent.side is not None:
            self.core.note_entry(intent.side, tick)
        return result

    def _harvest_outcomes(self, positions: list[PositionState]) -> int:
        open_ids = {p.position_id for p in positions}
        added = 0
        for record in self.store.confirmed_entry_intents_without_outcome():
            ticket = str(record["broker_position_id"])
            if ticket in open_ids:
                continue
            identifier = record["broker_identifier"]
            risk_amount = record["risk_amount"]
            entry_equity = record["entry_equity"]
            if identifier is None or risk_amount is None or entry_equity is None:
                self.store.append_event(time.time_ns(), "forward_evidence_incomplete",
                                        {"decision_id": record["decision_id"]})
                continue
            outcome = self.broker.closed_outcome(int(identifier))
            if outcome is None:
                continue
            saved = {
                "position_id": ticket, "decision_id": record["decision_id"],
                "broker_identifier": int(identifier), "model_generation": int(record["model_generation"]),
                "opened_ns": int(outcome["opened_ns"]), "closed_ns": int(outcome["closed_ns"]),
                "net_pnl": float(outcome["net_pnl"]), "profit": float(outcome["profit"]),
                "commission": float(outcome["commission"]), "swap": float(outcome["swap"]),
                "fee": float(outcome["fee"]), "volume": float(outcome["volume"]),
                "risk_amount": float(risk_amount), "entry_equity": float(entry_equity),
            }
            if self.store.record_trade_outcome(saved):
                added += 1
                self.store.append_event(
                    int(outcome["closed_ns"]), "forward_trade_closed",
                    {"decision_id": record["decision_id"], "position_id": ticket,
                     "model_generation": int(record["model_generation"]),
                     "net_pnl": float(outcome["net_pnl"]), "deal_ids": outcome["deal_ids"]},
                )
        if added and self.core.model.generation > 0:
            self.forward_report = self.forward.evaluate_store(
                self.store, self.core.model.generation, self.telemetry.summary())
            self.store.append_event(time.time_ns(), "forward_qualification_updated",
                                    asdict(self.forward_report))
        return added

    def _performance_multiplier(self) -> float:
        if self.core.model.generation <= 0:
            return 0.0
        if self.forward_report is None or self.forward_report.generation != self.core.model.generation:
            return 0.50
        return self.forward.performance_multiplier(self.forward_report)

    def _maybe_learn(self) -> dict | None:
        if self.tick_count == 0 or self.tick_count % self.settings.replay_every_ticks:
            return None
        replay = self.replay.build(self.store, extra_cost_spreads=self.quality.label_cost_spreads)
        previous_generation = self.core.model.generation
        cycle = self.learning.maybe_train()
        result = {"replay": asdict(replay),
                  "learning": {"attempted": cycle.attempted, "reason": cycle.reason}}
        if cycle.report is not None:
            result["learning"]["report"] = asdict(cycle.report)
        if self.core.model.generation != previous_generation:
            self.forward_report = None
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
        harvested = self._harvest_outcomes(positions)
        self.core.set_execution_context(
            size_multiplier=self.quality.size_multiplier,
            probability_penalty=self.quality.probability_penalty,
            performance_multiplier=self._performance_multiplier(),
            blocked=self.quality.blocked,
        )
        intents = self.core.on_tick(tick, positions)
        self.store.replace_positions(positions)
        halted = bool(self.store.meta("scalper_execution_halted", False))

        executed = 0
        if self.settings.execution_enabled and not halted:
            for idx, intent in enumerate(intents):
                if intent.kind is IntentKind.HOLD:
                    continue
                self._execute(tick, idx, intent, positions)
                executed += 1
                positions = self._positions()
            harvested += self._harvest_outcomes(positions)
        else:
            for intent in intents:
                if intent.kind is not IntentKind.HOLD:
                    self.store.append_event(
                        tick.ts_ns, "intent_observed_no_write",
                        {"kind": intent.kind.value, "reason": intent.reason,
                         "confidence": intent.confidence, "size": intent.size, "stop": intent.stop},
                    )

        learning = self._maybe_learn()
        return {
            "result": "processed", "tick_ns": tick.ts_ns, "positions": len(positions),
            "intents": len(intents), "executed": executed, "outcomes_harvested": harvested,
            "execution_enabled": self.settings.execution_enabled,
            "halted": bool(self.store.meta("scalper_execution_halted", False)),
            "model_generation": self.core.model.generation,
            "model_qualified": self.core.model.qualified, "learning": learning,
            "telemetry": self.telemetry.summary(), "execution_quality": self.quality.snapshot(),
            "forward_qualification": None if self.forward_report is None else asdict(self.forward_report),
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
