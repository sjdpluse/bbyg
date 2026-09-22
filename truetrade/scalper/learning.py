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
    champion_balanced_accuracy: float = 0.0
    challenger_balanced_accuracy: float = 0.0
    validation_long_fraction: float = 0.5
    validation_minority_count: int = 0
    selected_candidate: str = "global"
    reason: str = "criteria_not_met"


class ChampionChallenger:
    """Learning gate: training mutates challengers, never the serving champion.

    Two challengers may be evaluated on the same untouched chronological block:
    a long-horizon model trained on all eligible history and a recent-regime model
    trained only on the most recent eligible history. Only the better challenger may
    replace the champion, and only after class-aware validation gates pass.
    """

    def __init__(self, dimensions: int = 8):
        self.champion = OnlineLogit(dimensions)
        self.qualified = False
        self.generation = 0

    @staticmethod
    def _metrics(model: OnlineLogit, samples: list[Sample]) -> tuple[float, float, float]:
        if not samples:
            return float("inf"), 0.0, 0.0
        losses: list[float] = []
        correct = 0
        totals = {0: 0, 1: 0}
        correct_by_class = {0: 0, 1: 0}
        for s in samples:
            p = min(max(model.probability(s.x), 1e-9), 1 - 1e-9)
            losses.append(-(s.y * math.log(p) + (1 - s.y) * math.log(1 - p)))
            pred = 1 if p >= 0.5 else 0
            hit = int(pred == s.y)
            correct += hit
            totals[s.y] += 1
            correct_by_class[s.y] += hit
        recalls = [correct_by_class[y] / totals[y] for y in (0, 1) if totals[y] > 0]
        balanced = float(np.mean(recalls)) if recalls else 0.0
        return float(np.mean(losses)), correct / len(samples), balanced

    @staticmethod
    def _train(base: OnlineLogit, samples: list[Sample], epochs: int) -> OnlineLogit:
        challenger = base.clone()
        for _ in range(epochs):
            for sample in samples:
                challenger.update(sample)
        return challenger

    def fit_and_maybe_promote(
        self,
        train: list[Sample],
        validation: list[Sample],
        *,
        recent_train: list[Sample] | None = None,
        epochs: int = 4,
        min_train: int = 200,
        min_validation: int = 100,
        min_logloss_improvement: float = 0.01,
        min_accuracy: float = 0.53,
        min_balanced_accuracy: float = 0.52,
        min_validation_class_count: int = 20,
    ) -> PromotionReport:
        if set(map(id, train)) & set(map(id, validation)):
            raise ValueError("train and validation objects must be disjoint")
        if recent_train is not None and set(map(id, recent_train)) & set(map(id, validation)):
            raise ValueError("recent train and validation objects must be disjoint")
        if len(train) < min_train or len(validation) < min_validation:
            raise ValueError("insufficient independent learning data")
        if min_validation_class_count < 1:
            raise ValueError("minimum validation class count must be positive")

        global_candidate = self._train(self.champion, train, epochs)
        candidates: list[tuple[str, OnlineLogit]] = [("global", global_candidate)]
        if recent_train is not None and len(recent_train) >= min_train:
            candidates.append(("recent", self._train(self.champion, recent_train, epochs)))

        c_loss, c_acc, c_bal = self._metrics(self.champion, validation)
        evaluated: list[tuple[str, OnlineLogit, float, float, float]] = []
        for name, model in candidates:
            loss, acc, bal = self._metrics(model, validation)
            evaluated.append((name, model, loss, acc, bal))
        name, challenger, n_loss, n_acc, n_bal = min(
            evaluated,
            key=lambda item: (item[2], -item[4], -item[3]),
        )

        positives = sum(s.y for s in validation)
        negatives = len(validation) - positives
        minority = min(positives, negatives)
        long_fraction = positives / len(validation)
        class_usable = minority >= min_validation_class_count
        improve = c_loss - n_loss
        promoted = (
            class_usable
            and math.isfinite(n_loss)
            and improve >= min_logloss_improvement
            and n_acc >= min_accuracy
            and n_bal >= min_balanced_accuracy
            and n_bal >= c_bal
        )

        if not class_usable:
            reason = "validation_class_imbalance"
        elif promoted:
            reason = "promoted"
        elif not math.isfinite(n_loss):
            reason = "non_finite_challenger"
        elif improve < min_logloss_improvement:
            reason = "logloss_not_improved"
        elif n_bal < min_balanced_accuracy or n_bal < c_bal:
            reason = "balanced_accuracy_not_improved"
        elif n_acc < min_accuracy:
            reason = "accuracy_below_floor"
        else:
            reason = "criteria_not_met"

        if promoted:
            self.champion = challenger
            self.qualified = True
            self.generation += 1
        return PromotionReport(
            promoted=promoted,
            train_samples=len(train),
            validation_samples=len(validation),
            champion_logloss=c_loss,
            challenger_logloss=n_loss,
            champion_accuracy=c_acc,
            challenger_accuracy=n_acc,
            champion_balanced_accuracy=c_bal,
            challenger_balanced_accuracy=n_bal,
            validation_long_fraction=long_fraction,
            validation_minority_count=minority,
            selected_candidate=name,
            reason=reason,
        )

    def snapshot(self) -> dict:
        return {
            "qualified": bool(self.qualified),
            "generation": int(self.generation),
            "dimensions": int(len(self.champion.w)),
            "learning_rate": float(self.champion.lr),
            "l2": float(self.champion.l2),
            "weights": [float(x) for x in self.champion.w],
            "bias": float(self.champion.b),
        }

    def restore(self, document: dict) -> None:
        dimensions = int(document["dimensions"])
        weights = np.asarray(document["weights"], dtype=float)
        if weights.shape != (dimensions,) or not np.isfinite(weights).all():
            raise ValueError("invalid model snapshot")
        model = OnlineLogit(dimensions, float(document["learning_rate"]), float(document["l2"]))
        model.w = weights.copy()
        model.b = float(document["bias"])
        if not math.isfinite(model.b):
            raise ValueError("invalid model bias")
        self.champion = model
        self.qualified = bool(document["qualified"])
        self.generation = int(document["generation"])

    def probability_long(self, features: MicroFeatures) -> float:
        if not self.qualified:
            return 0.5
        return self.champion.probability(features.vector())
