# BBYG Phase 2 — local DEMO scalper runtime

Phase 2 moves the latency-sensitive loop next to MetaTrader 5 on Windows. Railway is not
part of the tick → decision → order path. The runtime is **DEMO-only** and refuses a real
account.

## Execution path

```text
MT5 tick
  -> durable tick journal
  -> causal microstructure features
  -> qualified champion model
  -> algorithmic position manager
  -> inventory/rate/spread risk gates
  -> durable execution intent
  -> exactly one MT5 order_send
  -> order/deal/position verification
  -> latency + slippage telemetry
```

There is no elapsed-time exit. Positions close or reduce because of market state: hard
risk, edge reversal, momentum/flow decay, or algorithmic profit protection. A broker-side
emergency stop remains mandatory so a terminal/process failure cannot leave an unlimited
position.

## Self-improvement

Ticks are stored locally in SQLite. Replay creates cost-aware labels from executable bid/ask
paths. If neither directional target resolves inside the evidence window, the row is left
unlabeled; the evidence window is **not** a trading time-stop.

Learning is champion/challenger:

1. training uses only older labeled rows;
2. a purge gap separates training from validation;
3. each chronological validation block is consumed once whether the challenger wins or loses;
4. a failed challenger never changes the serving model;
5. a promoted champion is persisted and restored after restart.

This design can improve with new data without repeatedly peeking at the same validation
window.

## Windows DEMO configuration

The original MT5 variables are reused for credentials. Do not put passwords in GitHub or
commit them to `.env` files.

```powershell
$env:MT5_MODE = "demo"
$env:MT5_LOGIN = "<demo trading login>"
$env:MT5_SERVER = "<exact demo server>"
$env:MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
$cred = Get-Credential -UserName $env:MT5_LOGIN -Message "MT5 DEMO password"
$env:MT5_PASSWORD = $cred.GetNetworkCredential().Password

$env:BBYG_MT5_SYMBOL = "XAUUSD"
$env:BBYG_STATE_DIR = "$PWD\data\bbyg-demo"
$env:BBYG_MT5_MAGIC = "731022"
$env:BBYG_POLL_INTERVAL_MS = "10"
$env:BBYG_MAX_SPREAD_POINTS = "50"
$env:BBYG_DEVIATION_POINTS = "20"
$env:BBYG_EMERGENCY_STOP_POINTS = "150"
$env:BBYG_RISK_FRACTION = "0.0025"
$env:BBYG_COMMISSION_PER_LOT = "0"

# Start in collect/learn mode. No order writes.
$env:BBYG_DEMO_EXECUTION = "false"
python -m scripts.bbyg_demo
```

After enough labeled data exists and a challenger qualifies, the model can emit entries.
To permit DEMO writes explicitly:

```powershell
$env:BBYG_DEMO_EXECUTION = "true"
python -m scripts.bbyg_demo
```

The adapter still refuses real accounts. Phase 2 is not a profitability claim and does not
unlock live execution.

## Failure semantics

BBYG persists a decision before a write. `order_send` is attempted once. If the response is
missing, malformed, interrupted, or cannot be reconciled to order/deal/position history, the
decision becomes `unknown` and execution is latched off. Restart does not blindly resend it.
Startup reconciliation checks the stable decision comment in MT5 history; unresolved writes
remain halted for investigation.

## Current conservative limits

The Phase 2 defaults intentionally start small: up to six BBYG positions, four in one
direction, 0.10 total lots, 0.08 directional lots, four order attempts per second, and a
maximum 0.5% risk budget per entry (default 0.25%). These are engineering guardrails, not
optimal trading parameters. Scale should be increased only after DEMO fill latency,
slippage, rejection rate, net expectancy, and drawdown are measured.
