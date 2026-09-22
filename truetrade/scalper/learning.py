from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import MicroFeatures


@dataclass(frozen=True)
class Sample:
    x: tuple[float, ...]
    y: int

    def __post_init__(self) -> None:
        if self.y not in (0, 1):
            raise ValueError("binary label required")
        if not self.x or not all(math.isfinite(v) for v in self.x):
            raise ValueError("finite feature vector required")


class OnlineLogit:
    """Small deterministic classifier suitable for local low-latency inference."""

    def __init__(self, dimensions: int = 8, learning_rate: float = 0.03, l2: float = 1e-4):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.w = np.zeros(dimensions, dtype=float)
        self.b = 0.0
        self.lr = float(learning_rate)
        self.l2 = float(l2)

    def clone(self) -> "OnlineLogit":
        other = OnlineLogit(len(self.w), self.lr, self.l2)
        other.w = self.w.copy()
        other.b = self.b
        return other

    def probability(self, x: tuple[float, ...]) -> float:
        a = np.asarray(x, dtype=float)
        if a.shape != self.w.shape or not np.isfinite(a).all():
            raise ValueError("invalid feature vector")
        z = float(np.clip(a @ self.w + self.b, -30, 30))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, sample: Sample) -> None:
        x = np.asarray(sample.x, dtype=float)
        p = self.probability(sample.x)
        error = p - sample.y
        self.w -= self.lr * (error * x + self.l2 * self.w)
        self.b -= self.lr * error
        self.w = np.clip(self.w, -8.0, 8.0)
        self.b = float(np.clip(self.b, -8.0, 8.0))


@dataclass(frozen=True)
class PromotionReport:
    promoted: bool
    train_samples: int
    validation_samples: int
    champion_logloss: float
    challenger_logloss: float
    champion_accuracy: float
    challenger_accuracy: float


class ChampionChallenger:
    """Learning gate: training mutates a challenger, never the serving champion.

    Promotion requires genuinely separate validation samples and improvement in both
    log-loss and directional accuracy. This prevents uncontrolled online self-modification.
    """

    def __init__(self, dimensions: int = 8):
        self.champion = OnlineLogit(dimensions)
        self.qualified = False
        self.generation = 0

    @staticmethod
    def _metrics(model: OnlineLogit, samples: list[Sample]) -> tuple[float, float]:
        if not samples:
            return float("inf"), 0.0
        losses = []
        correct = 0
        for s in samples:
            p = min(max(model.probability(s.x), 1e-9), 1 - 1e-9)
            losses.append(-(s.y * math.log(p) + (1 - s.y) * math.log(1 - p)))
            correct += int((p >= 0.5) == bool(s.y))
        return float(np.mean(losses)), correct / len(samples)

    def fit_and_maybe_promote(
        self,
        train: list[Sample],
        validation: list[Sample],
        *,
        epochs: int = 4,
        min_train: int = 200,
        min_validation: int = 100,
        min_logloss_improvement: float = 0.01,
        min_accuracy: float = 0.53,
    ) -> PromotionReport:
        if set(map(id, train)) & set(map(id, validation)):
            raise ValueError("train and validation objects must be disjoint")
        if len(train) < min_train or len(validation) < min_validation:
            raise ValueError("insufficient independent learning data")
        challenger = self.champion.clone()
        for _ in range(epochs):
            for s in train:
                challenger.update(s)

        c_loss, c_acc = self._metrics(self.champion, validation)
        n_loss, n_acc = self._metrics(challenger, validation)
        improve = c_loss - n_loss
        promoted = (
            math.isfinite(n_loss)
            and improve >= min_logloss_improvement
            and n_acc >= min_accuracy
            and n_acc >= c_acc
        )
        if promoted:
            self.champion = challenger
            self.qualified = True
            self.generation += 1
        return PromotionReport(promoted, len(train), len(validation), c_loss, n_loss, c_acc, n_acc)

    def probability_long(self, features: MicroFeatures) -> float:
        if not self.qualified:
            return 0.5
        return self.champion.probability(features.vector())
