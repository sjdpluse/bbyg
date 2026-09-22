# BBYG Phase 3 — adaptive DEMO scalper

Phase 3 turns the Phase-2 tick engine into an adaptive, auditable DEMO system. It does **not** enable live trading and it does not claim profitability.

## What changed

### 1. Adaptive position sizing

`AdaptiveSizer` sizes an entry from model confidence, trend efficiency, micro-volatility, spread regime, measured execution quality, forward-demo performance, and remaining inventory capacity. The MT5 adapter remains the final authority and independently checks the broker-side emergency-stop risk budget and symbol volume rules.

### 2. Portfolio exposure engine

`PortfolioExposureEngine` measures gross and directional inventory. If recovered or externally changed state is already outside configured limits, it emits reduction intents before considering new entries. `RiskController` still blocks new entries when total, directional, position-count, spread, or order-rate limits are reached.

### 3. Market-state protection, not a time stop

There is no elapsed-time exit. `DynamicProtectionManager` can tighten the broker-side stop after favorable excursion. Strong edge permits a looser trail so a winner can continue; weakening edge tightens protection. Stops may tighten only; the MT5 adapter refuses widening.

`AlgorithmicExitManager` can also scale out part of a profitable position when edge decays, close on reversal, close on hard adverse risk, or close after meaningful profit giveback.

### 4. Execution-quality feedback

Every verified market fill records latency and adverse slippage. A durable EWMA produces an entry-probability penalty, a size multiplier, an execution-quality block under severely degraded conditions, and a conservative extra-cost term for future tick labels. Training therefore does not continue assuming ideal fills when real DEMO execution shows worse costs.

### 5. Durable closed-trade evidence

Each confirmed entry stores the model generation, exact decision ID, MT5 position identifier, entry equity, and actual emergency-stop risk amount. Once that position is fully closed, MT5 deal history is harvested into a durable `trade_outcome` with profit, commission, swap, fee, volume, timestamps, and net PnL.

### 6. Forward-demo qualification

`ForwardDemoQualifier` evaluates each model generation separately. Default evidence requires at least:

- 100 unique closed DEMO trades;
- 30 calendar days;
- positive net PnL;
- profit factor >= 1.20;
- max drawdown <= 10%;
- positive lower-95% bootstrap expectancy in R;
- qualified p95 execution latency and slippage.

A pass means only `forward-demo qualified`. It never enables live trading.

### 7. Crash recovery

The execution journal distinguishes `created`, `submitted`, `confirmed`, `rejected`, and `unknown` states. A crash before submission is safely rejected on restart. Market writes are reconciled from the exact MT5 decision comment/deal history. Protection updates are reconciled from broker stop readback. An unresolved write leaves execution halted. There is no blind order retry after an uncertain write.

## Runtime path

```text
MT5 DEMO tick
    ↓
durable tick journal
    ↓
causal microstructure features
    ↓
qualified champion model
    ↓
algorithmic exits + portfolio guard
    ↓
dynamic broker protection
    ↓
execution-quality adjusted entry threshold
    ↓
adaptive sizing
    ↓
local MT5 write + verification
    ↓
deal-history outcome harvest
    ↓
forward DEMO evidence
    ↓
cost-aware replay
    ↓
challenger training
    ↓
independent validation
    ↓
champion promotion only if improved
```

## Run

Keep execution disabled while collecting ticks and training:

```powershell
$env:MT5_MODE = "demo"
$env:BBYG_DEMO_EXECUTION = "false"
python -m scripts.bbyg_demo
```

After a champion is qualified and the dedicated DEMO account is ready, execution can be explicitly enabled:

```powershell
$env:BBYG_DEMO_EXECUTION = "true"
python -m scripts.bbyg_demo
```

The program still refuses a non-DEMO MT5 account.

## Important separation

Historical label lookahead is used only to decide whether a past feature row has enough evidence to be a training example. It is not a position timeout. Runtime position management never closes a trade because a fixed number of seconds elapsed.
