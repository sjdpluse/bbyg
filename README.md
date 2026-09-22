# BBYG

BBYG is a local, tick-driven XAUUSD scalping research and **DEMO execution** system built from the Babayaga MT5 safety foundation.

It is designed around three rules:

1. market decisions are made locally next to MT5 rather than through a cloud round-trip;
2. positions are managed from current market state, not a fixed holding-time timer;
3. learning uses champion/challenger promotion and forward DEMO evidence instead of uncontrolled self-modification.

## Current architecture

```text
MT5 DEMO
  ↓
tick ingestion
  ↓
SQLite/WAL durable journal
  ↓
microstructure features
  ↓
qualified champion model
  ↓
algorithmic entry / exit / scale-out / dynamic protection
  ↓
adaptive sizing + portfolio exposure limits
  ↓
local MT5 order router
  ↓
fill verification + latency/slippage telemetry
  ↓
closed-trade outcome harvest
  ↓
forward DEMO qualification
  ↓
cost-aware replay + challenger training
```

The serving model does not train itself in place. New data trains a challenger; a separate chronological validation block must show improvement before promotion. Validation blocks are consumed once to reduce repeated peeking.

## Exit logic

BBYG has no fixed time stop. A position can remain open while the measured edge remains favorable. Possible actions include:

- hold while edge/momentum remain intact;
- tighten broker-side protection after favorable excursion;
- partial profit scale-out as edge decays;
- full close on edge reversal;
- full close on hard adverse risk;
- profit-lock close after meaningful giveback.

A profitable close is never guaranteed. The system must be able to accept a controlled loss when the market invalidates the trade.

## Safety scope

- Execution adapter is DEMO-only.
- Non-DEMO MT5 accounts are rejected.
- Market writes have exactly one send attempt.
- Uncertain outcomes halt execution instead of blind retry.
- A dedicated hedging account is required.
- Foreign/manual XAUUSD positions and pending orders block new BBYG entries.
- Every entry has a broker-side emergency stop.
- Forward qualification does not enable live trading.

## Run tests

```bash
python -m unittest discover -s tests -v
```

CI runs on Windows and Linux.

## Run the local DEMO process

Install the MetaTrader5 Python package on Windows and set the exact DEMO login, trading password, server, and terminal path.

Collection/training without writes:

```powershell
$env:MT5_MODE = "demo"
$env:BBYG_DEMO_EXECUTION = "false"
python -m scripts.bbyg_demo
```

Explicit DEMO execution:

```powershell
$env:BBYG_DEMO_EXECUTION = "true"
python -m scripts.bbyg_demo
```

See:

- `docs/SCALPER_ARCHITECTURE.md`
- `docs/SCALPER_PHASE2.md`
- `docs/SCALPER_PHASE3.md`

Legacy Babayaga modules remain in the repository because BBYG reuses its tested MT5 and research foundations, but the `truetrade/scalper/` package is the dedicated BBYG path.
