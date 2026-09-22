"""BBYG tick-level scalping research and DEMO execution core."""

from .engine import EntrySettings, ScalperCore
from .execution import (
    DemoExecutionError, DemoMT5Execution, DemoMT5Settings,
    ExecutionRejected, ExecutionResult, ExecutionUncertain,
)
from .features import TickFeatureEngine
from .labels import CostAwareLabeler, LabelOutcome, LabelSettings
from .learning import ChampionChallenger, OnlineLogit, PromotionReport, Sample
from .position import AlgorithmicExitManager, ExitSettings
from .replay import ReplayReport, TickReplayBuilder
from .risk import RiskController, ScalperRiskLimits
from .runtime import DemoScalperRuntime, RuntimeSettings
from .store import ScalperStore, StoredSample
from .telemetry import ExecutionObservation, ExecutionTelemetry
from .trainer import LearningCycle, LearningSettings, SelfImprovementController
from .types import Intent, IntentKind, MicroFeatures, PositionState, Side, Tick

__all__ = [
    "AlgorithmicExitManager", "ChampionChallenger", "CostAwareLabeler",
    "DemoExecutionError", "DemoMT5Execution", "DemoMT5Settings", "DemoScalperRuntime",
    "EntrySettings", "ExecutionObservation", "ExecutionRejected", "ExecutionResult",
    "ExecutionTelemetry", "ExecutionUncertain", "ExitSettings", "Intent", "IntentKind",
    "LabelOutcome", "LabelSettings", "LearningCycle", "LearningSettings", "MicroFeatures",
    "OnlineLogit", "PositionState", "PromotionReport", "ReplayReport", "RiskController",
    "RuntimeSettings", "Sample", "ScalperCore", "ScalperRiskLimits", "ScalperStore",
    "SelfImprovementController", "Side", "StoredSample", "Tick", "TickFeatureEngine",
    "TickReplayBuilder",
]
