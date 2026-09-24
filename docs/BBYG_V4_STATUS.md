# BBYG v4 Build Status

Updated: 2026-09-24

## Current decision

v2/v3 autonomous DEMO runners are retained as failed/negative experimental evidence and are not the development target.

BBYG v4 will not be allowed to autonomously execute MT5 orders until it passes the following gates in order:

1. offline replay correctness;
2. walk-forward economic validation;
3. live shadow stability;
4. single-position DEMO qualification;
5. limited multi-position DEMO qualification.

More position fanout is explicitly not treated as a substitute for predictive edge.

## Repository correction

The repository **does contain PPO** under `truetrade/rl/ppo.py` and tests for it under `tests/test_learning.py`. Earlier v4 design notes stating that PPO was absent were incorrect. PPO is therefore an existing research component, but it will not be wired into v4 execution until the state encoder, reward contract and memory evaluation are stable.

## Implemented v4 foundation

### Architecture

- `docs/BBYG_V4_CONNECTOME_ARCHITECTURE.md`

### Persistent experiential memory

- `truetrade/scalper/experience.py`
- durable `market_episodes` table;
- causal state embeddings;
- executed and counterfactual LONG/SHORT outcomes;
- cosine similarity recall;
- memory confidence;
- reward-modulated memory signal;
- bounded artificial-dopamine reward helper.

### Trade thesis engine

- `truetrade/scalper/thesis.py`
- CANDIDATE -> CONFIRMED -> ENTERED -> DEVELOPING/PROTECTED -> EXITING -> CLOSED;
- persistent evidence requirement before entry;
- memory disagreement can block entry;
- entry and exit hysteresis;
- repeated exit confirmation requirement;
- no single prediction flip can create a valid thesis transition by itself.

### Tests

- `tests/test_v4_memory_thesis.py`
- reward bounds;
- similarity recall;
- delayed episode outcome resolution;
- persistent entry evidence;
- persistent/aged reversal exit;
- memory disagreement blocking.

## Next implementation order

### Phase 1A — market state encoder

Create one causal state vector joining:

- existing tick microstructure;
- 1s / 5s / 1m bars;
- EMA slope and separation;
- RSI;
- ATR / realized volatility;
- local structure / breakout distance;
- spread regime;
- session / time features;
- execution-quality context.

The encoder must be deterministic and replay/live-parity tested.

### Phase 1B — counterfactual episode resolver

Every eligible state, including FLAT decisions, receives future counterfactual LONG and SHORT rewards using executable bid/ask paths and costs. This prevents learning only from trades the current policy happened to execute.

### Phase 2 — thesis coordinator

Combine model probability, technical evidence, episodic recall, regime state and optional fundamental context into one durable `TradeThesis`. No order proposal exists until the thesis is confirmed.

### Phase 3 — connectome substrate

Implement two controlled memory candidates:

1. mushroom-body-inspired sparse associative network;
2. MaleCNS-derived sparse recurrent reservoir.

Also implement matched random-graph controls. A connectome model survives only if it beats simpler memory baselines on untouched forward data.

### Phase 4 — sequence model

Build a temporal encoder (TCN first, Transformer second if justified) over multi-timescale market state. Compare:

- sequence only;
- sequence + ordinary reservoir;
- sequence + episodic recall;
- sequence + connectome memory.

### Phase 5 — PPO integration

Reuse the repository's existing auditable PPO implementation only after the environment, observation schema and reward contract are frozen. PPO actions become WAIT / OPEN_LONG / OPEN_SHORT / HOLD / ADD / REDUCE / CLOSE and remain behind risk/qualification gates.

### Phase 6 — shadow runtime

Run live with zero MT5 writes. Persist every proposal, thesis transition, memory recall, counterfactual outcome and model generation.

### Phase 7 — controlled DEMO

Begin with one position. Increase concurrency only after measured forward evidence remains positive after executable costs.

## Non-negotiable scientific rules

- no using unresolved/future information in features;
- no training on validation blocks;
- no promotion from repeated peeking at the same validation period;
- no learning only from executed trades;
- no claim that biological topology is useful until it beats matched controls;
- no autonomous live-money trading;
- no automatic scaling of position count based on confidence alone.
