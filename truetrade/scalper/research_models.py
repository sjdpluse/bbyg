from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "RobustScaler":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("finite 2D training matrix required")
        center = np.median(x, axis=0)
        q25 = np.quantile(x, 0.25, axis=0)
        q75 = np.quantile(x, 0.75, axis=0)
        robust = (q75 - q25) / 1.349
        std = np.std(x, axis=0)
        scale = np.where(robust > 1e-6, robust, np.where(std > 1e-6, std, 1.0))
        return cls(center=center.astype(float), scale=scale.astype(float))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.center) or not np.isfinite(x).all():
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.center) / self.scale, -8.0, 8.0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def quadratic_expand(x: np.ndarray) -> np.ndarray:
    """Return original standardized features plus all x_i*x_j interactions for i<=j."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite 2D feature matrix required")
    cols = [x]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            cols.append((x[:, i] * x[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary labels required")
    n = len(y)
    positive = max(float(y.sum()), 1.0)
    negative = max(float(n - y.sum()), 1.0)
    return np.where(y > 0.5, n / (2.0 * positive), n / (2.0 * negative))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("aligned finite labels/probabilities required")
    p = np.clip(p, EPS, 1.0 - EPS)
    pred = (p >= 0.5).astype(int)
    loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    accuracy = float(np.mean(pred == y))
    recalls: list[float] = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == cls)))
    balanced = float(np.mean(recalls)) if recalls else 0.0
    confidence = np.maximum(p, 1.0 - p)

    def selective(threshold: float) -> tuple[float, float | None]:
        mask = confidence >= threshold
        if not mask.any():
            return 0.0, None
        return float(np.mean(mask)), float(np.mean(pred[mask] == y[mask]))

    cov55, acc55 = selective(0.55)
    cov60, acc60 = selective(0.60)
    return {
        "logloss": loss,
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "actual_long_fraction": float(np.mean(y)),
        "predicted_long_probability": float(np.mean(p)),
        "probability_std": float(np.std(p)),
        "calibration_bias": float(np.mean(p) - np.mean(y)),
        "confidence_55_coverage": cov55,
        "confidence_55_accuracy": acc55,
        "confidence_60_coverage": cov60,
        "confidence_60_accuracy": acc60,
    }


@dataclass(frozen=True)
class LogitModel:
    weights: np.ndarray
    bias: float

    def probability(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(x, dtype=float) @ self.weights + self.bias)


def fit_logit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int = 160,
    learning_rate: float = 0.06,
    l2: float = 1e-3,
    balanced: bool = True,
) -> LogitModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("invalid logistic training data")
    if iterations < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid logistic optimizer settings")
    weights = class_weights(y) if balanced else np.ones(len(y), dtype=float)
    weight_sum = float(weights.sum())
    w = np.zeros(x.shape[1], dtype=float)
    b = 0.0
    for _ in range(iterations):
        p = sigmoid(x @ w + b)
        err = (p - y) * weights
        grad_w = (x.T @ err) / weight_sum + l2 * w
        grad_b = float(err.sum() / weight_sum)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        w = np.clip(w, -8.0, 8.0)
        b = float(np.clip(b, -8.0, 8.0))
    return LogitModel(w, b)


@dataclass(frozen=True)
class MLPModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float

    def probability(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return sigmoid(hidden @ self.w2 + self.b2)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_units: int = 16,
    iterations: int = 180,
    learning_rate: float = 0.01,
    l2: float = 1e-3,
    seed: int = 731022,
) -> MLPModel:
    """Deterministic one-hidden-layer tanh MLP trained with full-batch Adam.

    The model is research-only. Validation data is never used by this optimizer.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("invalid MLP training data")
    if not 2 <= hidden_units <= 128 or iterations < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid MLP optimizer settings")

    rng = np.random.default_rng(int(seed))
    input_dim = x.shape[1]
    limit1 = math.sqrt(6.0 / (input_dim + hidden_units))
    w1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_units))
    b1 = np.zeros(hidden_units, dtype=float)
    limit2 = math.sqrt(6.0 / (hidden_units + 1))
    w2 = rng.uniform(-limit2, limit2, size=hidden_units)
    b2 = 0.0

    sample_weights = class_weights(y)
    weight_sum = float(sample_weights.sum())
    beta1, beta2 = 0.9, 0.999
    adam_eps = 1e-8
    m_w1 = np.zeros_like(w1)
    v_w1 = np.zeros_like(w1)
    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)
    m_w2 = np.zeros_like(w2)
    v_w2 = np.zeros_like(w2)
    m_b2 = 0.0
    v_b2 = 0.0

    for step in range(1, iterations + 1):
        hidden = np.tanh(x @ w1 + b1)
        p = sigmoid(hidden @ w2 + b2)
        dz2 = (p - y) * sample_weights / weight_sum
        grad_w2 = hidden.T @ dz2 + l2 * w2
        grad_b2 = float(dz2.sum())
        dh = dz2[:, None] * w2[None, :]
        dz1 = dh * (1.0 - hidden * hidden)
        grad_w1 = x.T @ dz1 + l2 * w1
        grad_b1 = dz1.sum(axis=0)

        for param, grad, m, v in (
            (w1, grad_w1, m_w1, v_w1),
            (b1, grad_b1, m_b1, v_b1),
            (w2, grad_w2, m_w2, v_w2),
        ):
            m *= beta1
            m += (1.0 - beta1) * grad
            v *= beta2
            v += (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)

        m_b2 = beta1 * m_b2 + (1.0 - beta1) * grad_b2
        v_b2 = beta2 * v_b2 + (1.0 - beta2) * (grad_b2 * grad_b2)
        m_hat_b2 = m_b2 / (1.0 - beta1 ** step)
        v_hat_b2 = v_b2 / (1.0 - beta2 ** step)
        b2 -= learning_rate * m_hat_b2 / (math.sqrt(v_hat_b2) + adam_eps)

        w1 = np.clip(w1, -8.0, 8.0)
        b1 = np.clip(b1, -8.0, 8.0)
        w2 = np.clip(w2, -8.0, 8.0)
        b2 = float(np.clip(b2, -8.0, 8.0))

    return MLPModel(w1, b1, w2, b2)


def fit_predict_architecture(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    linear_iterations: int = 160,
    mlp_iterations: int = 180,
    seed: int = 731022,
) -> np.ndarray:
    scaler = RobustScaler.fit(train_x)
    tx = scaler.transform(train_x)
    vx = scaler.transform(validation_x)

    if architecture == "linear":
        model = fit_logit(tx, train_y, iterations=linear_iterations, balanced=True)
        return model.probability(vx)
    if architecture == "quadratic":
        qtx = quadratic_expand(tx)
        qvx = quadratic_expand(vx)
        model = fit_logit(qtx, train_y, iterations=linear_iterations, learning_rate=0.035,
                          l2=2e-3, balanced=True)
        return model.probability(qvx)
    if architecture == "mlp":
        model = fit_mlp(tx, train_y, iterations=mlp_iterations, seed=seed)
        return model.probability(vx)
    raise ValueError(f"unknown research architecture: {architecture}")
