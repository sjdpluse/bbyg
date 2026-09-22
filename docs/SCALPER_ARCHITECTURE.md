# BBYG scalper architecture — Phase 1

BBYG is a tick-driven, local-first scalping system derived from Babayaga's execution-safety foundation.
Phase 1 is deliberately broker-neutral and emits **intents**, not MT5 writes. Live trading remains disabled.

## Core principles

1. **No fixed time exit.** Elapsed time is never by itself a close condition. A position remains open while its measured directional edge, micro-momentum and risk state justify holding it.
2. **Algorithmic exits.** Positions close on edge reversal, hard adverse movement, or a market-state profit lock after favorable excursion and meaningful giveback. Partial reduction is allowed when a profitable edge decays without reversing.
3. **Local tick loop.** Latency-sensitive inference and position management belong beside MT5 on Windows/VPS. Railway remains suitable for model training, registry, metrics and monitoring, not the per-tick execution round trip.
4. **Multi-position inventory with limits.** The core permits several positions but caps total positions, same-direction inventory, size and order rate.
5. **Learning is champion/challenger, not uncontrolled self-editing.** New observations train a challenger only. The serving champion changes only after separate validation data show improvement in log-loss and directional accuracy.
6. **No blind broker retries.** The existing Babayaga certainty/journal design remains the execution standard for later MT5 integration.

## Tick feature layer

`truetrade.scalper.features.TickFeatureEngine` creates causal, dimensionless microstructure features:

- robust spread z-score;
- fast and slow price velocity in spread-units/second;
- acceleration;
- directional tick imbalance;
- signed trend efficiency;
- realized micro-volatility relative to spread;
- last move relative to spread;
- raw local noise estimate.

Normalizing directional movement by spread is intentional: an apparent price move that is smaller than transaction cost should not look equally attractive to the model.

## Learning loop

The first learning primitive is a small low-latency logistic classifier. It is not automatically trusted.
A challenger trains on one chronological sample set and must improve on a **disjoint** validation set before promotion.
Future phases will persist datasets, enforce chronological walk-forward splits, add transaction-cost-aware labels, and compare challengers against the incumbent on untouched windows.

## Exit state machine

There is no timer branch:

```text
OPEN -> HOLD while edge remains valid
     -> REDUCE when profitable but edge/momentum materially decays
     -> CLOSE on edge reversal
     -> CLOSE on hard adverse movement
     -> CLOSE on algorithmic profit lock after favorable excursion + giveback
```

A profitable position can therefore remain open for as long as the market state supports it.

## Phase 2

- MT5 tick collector using local `symbol_info_tick`/tick history;
- durable tick journal and replay format;
- transaction-cost-aware outcome labeling;
- local demo execution adapter with unique decision IDs and exact fill reconciliation;
- per-position ownership/magic tracking for multi-position hedging accounts;
- latency, slippage, rejection and fill-quality telemetry;
- walk-forward challenger promotion and rollback;
- demo-only load tests before any consideration of larger concurrency.
