# BBYG v4 — Neuromorphic Adaptive Trading Architecture

Status: **research design only**. No execution authorization.

## Why v4 exists

The experimental v2/v3 runners exposed a structural failure: they could create and liquidate many positions from short-horizon score fluctuations without a durable market memory, independent evidence accumulation, or sufficiently slow policy adaptation. High fanout amplified weak decisions rather than improving decision quality.

v4 therefore separates **