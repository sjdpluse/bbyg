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

### Phase 1A — causal multi-timescale state encoder

Implemented in `truetrade/scalper/state_encoder.py`.

The encoder now joins, causally and incrementally:

- existing tick microstructure;
- partial/current 1-second, 5-second and 60-second bars;
- returns;
- EMA 9 / EMA 21 distances and separation;
- EMA slope;
- RSI-14;
- ATR-14;
- realized volatility;
- local range position;
- breakout distance;
- relative volume;
- UTC cyclical time features;
- spread fraction;
- optional execution latency/slippage/failure context.

The state encoder also classifies a coarse causal regime:

- `quiet`;
- `range`;
- `trend_up`;
- `trend_down`;
- `shock`.

Important parity rule: replay and live must both feed sequential ticks through the same encoder. Missing buckets are not fabricated and the current partial bar contains only observations already received.

The encoder can now also advance its internal causal bars with `compute=False`; this allows strided offline replay without skipping state updates or materializing every embedding.

### Phase 1B — counterfactual outcome resolver

Implemented in `truetrade/scalper/counterfactual.py`.

Every eligible anchor can now be evaluated independently of the policy for both hypothetical directions:

- executable LONG from the first strictly later ask;
- executable SHORT from the first strictly later bid;
- fixed causal future horizon;
- MFE and MAE in R units;
- terminal net R after explicit spread-equivalent cost;
- bounded reward using the same artificial-dopamine reward components as experiential memory.

This means future learning no longer has to depend only on trades the current policy chose to execute. FLAT/rejected states can also become training evidence after their future path resolves.

### Phase 1C — offline episode builder and replay parity audit

Implemented in:

- `truetrade/scalper/offline_episodes.py`;
- `scripts/bbyg_v4_build_episodes.py`;
- `tests/test_v4_offline_episodes.py`.

The builder now:

- replays persisted MT5 ticks through the exact same v4 state encoder;
- advances all intermediate ticks causally while only materializing states at the frozen stride;
- resets state across market gaps larger than the configured boundary;
- refuses counterfactual horizons that cross a market gap;
- computes policy-independent LONG and SHORT future outcomes;
- persists FLAT/rejected-style `MarketEpisode` evidence before any policy exists;
- stores LONG/SHORT net R, MFE, MAE and artificial-dopamine rewards;
- uses a vectorized/chunked future-path resolver for offline scale;
- writes a frozen dataset signature based on the encoder schema and outcome contract;
- writes a deterministic binary SHA-256 episode digest;
- independently replays the same source ticks during audit and requires exact signature, digest and episode-count parity.

The CLI is deliberately write-safe. Dataset replacement requires both `--build` and `--yes-replace`. Audit mode performs no episode writes.

Example local build:

```powershell
python -m scripts.bbyg_v4_build_episodes --build --yes-replace
```

Independent parity replay:

```powershell
python -m scripts.bbyg_v4_build_episodes --audit
```

A parity audit failure is a hard stop before model training.

### Tests

- `tests/test_v4_memory_thesis.py`
- `tests/test_v4_state_counterfactual.py`
- `tests/test_v4_offline_episodes.py`
- reward bounds;
- similarity recall;
- delayed episode outcome resolution;
- persistent entry evidence;
- persistent/aged reversal exit;
- memory disagreement blocking;
- deterministic multi-timescale state encoding;
- strict chronological tick enforcement;
- counterfactual LONG/SHORT resolution from one identical future path;
- first-strictly-later executable entry semantics;
- deterministic offline dataset digest;
- exact second-replay parity;
- gap segmentation;
- digest sensitivity to source-path changes.

## Next implementation order

### Phase 1D — dataset diagnostics and frozen research split

Next:

- run Phase 1C against the real persisted XAUUSD tick database;
- require exact build/audit parity;
- profile regime coverage and reward distributions;
- inspect directional imbalance and pathological reward tails;
- freeze chronological train / calibration / validation boundaries;
- persist a dataset manifest containing tick range, episode count, signature and digest;
- reject model training if coverage or parity gates fail.

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
