from __future__ import annotations

from dataclasses import asdict, dataclass

from .learning import ChampionChallenger, PromotionReport
from .sample_intervals import load_label_intervals
from .store import ScalperStore


@dataclass(frozen=True)
class LearningSettings:
    min_train_samples: int = 400
    validation_block: int = 120
    purge_samples: int = 40
    recent_train_samples: int = 5000
    min_logloss_improvement: float = 0.005
    min_accuracy: float = 0.53
    min_balanced_accuracy: float = 0.52
    min_validation_class_count: int = 20
    epochs: int = 4

    def __post_init__(self) -> None:
        if self.min_train_samples < 200 or self.validation_block < 100 or self.purge_samples < 1:
            raise ValueError("learning windows too small")
        if self.recent_train_samples < self.min_train_samples:
            raise ValueError("recent training window too small")
        if not 0.5 <= self.min_accuracy <= 1 or not 0.5 <= self.min_balanced_accuracy <= 1:
            raise ValueError("invalid accuracy floors")
        if self.min_validation_class_count < 1:
            raise ValueError("minimum validation class count must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class LearningCycle:
    attempted: bool
    reason: str
    report: PromotionReport | None = None
    validation_end_id: int | None = None


class SelfImprovementController:
    """Consumes each chronological validation block once, promoted or not."""

    META_LAST_VALIDATION = "scalper_last_validation_sample_id"
    META_MODEL = "scalper_champion_snapshot"

    def __init__(self, store: ScalperStore, learner: ChampionChallenger,
                 settings: LearningSettings | None = None):
        self.store = store
        self.learner = learner
        self.settings = settings or LearningSettings()
        saved = store.meta(self.META_MODEL)
        if saved:
            learner.restore(saved)

    def maybe_train(self) -> LearningCycle:
        cfg = self.settings
        rows = self.store.samples()
        if not rows:
            return LearningCycle(False, "no_labeled_samples")
        last_consumed = int(self.store.meta(self.META_LAST_VALIDATION, 0) or 0)
        fresh = [r for r in rows if r.sample_id > last_consumed]
        if len(fresh) < cfg.validation_block:
            return LearningCycle(False, "waiting_for_new_validation_block")

        validation_rows = (rows[-cfg.validation_block:] if last_consumed == 0
                           else fresh[: cfg.validation_block])
        validation_start_id = validation_rows[0].sample_id
        validation_start_ts = validation_rows[0].feature_ts_ns
        train_rows = [r for r in rows if r.sample_id < validation_start_id]

        # Prefer exact label-resolution intervals so no training label is allowed to
        # inspect a tick at or beyond the validation feature boundary.  The fixed
        # sample purge remains a compatibility fallback for old/synthetic datasets.
        intervals = load_label_intervals(self.store)
        if intervals and all(r.feature_ts_ns in intervals for r in train_rows):
            before = len(train_rows)
            train_rows = [
                r for r in train_rows
                if intervals[r.feature_ts_ns].label_end_ts_ns < validation_start_ts
            ]
            purged = before - len(train_rows)
            purge_mode = "label_interval"
        else:
            if len(train_rows) <= cfg.purge_samples:
                return LearningCycle(False, "insufficient_pre_validation_history")
            train_rows = train_rows[: -cfg.purge_samples]
            purged = cfg.purge_samples
            purge_mode = "fixed_sample_fallback"

        if len(train_rows) < cfg.min_train_samples:
            return LearningCycle(False, "insufficient_training_history")

        recent_rows = train_rows[-cfg.recent_train_samples:]
        report = self.learner.fit_and_maybe_promote(
            [r.sample for r in train_rows],
            [r.sample for r in validation_rows],
            recent_train=[r.sample for r in recent_rows],
            epochs=cfg.epochs,
            min_train=cfg.min_train_samples,
            min_validation=cfg.validation_block,
            min_logloss_improvement=cfg.min_logloss_improvement,
            min_accuracy=cfg.min_accuracy,
            min_balanced_accuracy=cfg.min_balanced_accuracy,
            min_validation_class_count=cfg.min_validation_class_count,
        )
        end_id = validation_rows[-1].sample_id
        self.store.set_meta(self.META_LAST_VALIDATION, end_id)
        self.store.append_event(
            validation_rows[-1].feature_ts_ns,
            "learning_validation_consumed",
            {
                **asdict(report),
                "validation_start_id": validation_start_id,
                "validation_end_id": end_id,
                "recent_train_samples": len(recent_rows),
                "purge_mode": purge_mode,
                "purged_training_samples": purged,
            },
        )
        if report.promoted:
            self.store.set_meta(self.META_MODEL, self.learner.snapshot())
            self.store.append_event(
                validation_rows[-1].feature_ts_ns,
                "model_promoted",
                {"generation": self.learner.generation, "validation_end_id": end_id,
                 "selected_candidate": report.selected_candidate},
            )
            return LearningCycle(True, "challenger_promoted", report, end_id)
        return LearningCycle(True, report.reason or "challenger_rejected", report, end_id)
