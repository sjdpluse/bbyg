"""BBYG tick-level scalping research and DEMO execution core."""

from .engine import EntrySettings, ScalperCore
from .execution import (
    DemoExecutionError, DemoMT5Execution, DemoMT5Settings,
    ExecutionRejected, ExecutionResult, ExecutionUncertain,
)
from .features import TickFeatureEngine
from .forward import ForwardDemoQualifier, ForwardQualificationReport, ForwardQualificationSettings
from .labels import CostAwareLabeler, LabelOutcome, LabelSettings
from .learning import ChampionChallenger, OnlineLogit, PromotionReport, Sample
from .portfolio import PortfolioExposureEngine, PortfolioSnapshot
from .position import AlgorithmicExitManager, ExitSettings
from .protection import DynamicProtectionManager, ProtectionSettings
from .quality import ExecutionQualityController, ExecutionQualitySettings
from .replay import ReplayReport, TickReplayBuilder
from .risk import RiskController, ScalperRiskLimits
from .runtime import DemoScalperRuntime, RuntimeSettings
from .sizing import AdaptiveSizer, AdaptiveSizeSettings
from .store import ScalperStore, StoredSample
from .telemetry import ExecutionObservation, ExecutionTelemetry
from .trainer import LearningCycle, LearningSettings, SelfImprovementController
from .types import Intent, IntentKind, MicroFeatures, PositionState, Side, Tick

__all__ = [
    "AdaptiveSizer", "AdaptiveSizeSettings", "AlgorithmicExitManager",
    "ChampionChallenger", "CostAwareLabeler", "DemoExecutionError",
    "DemoMT5Execution", "DemoMT5Settings", "DemoScalperRuntime",
    "DynamicProtectionManager", "EntrySettings", "ExecutionObservation",
    "ExecutionQualityController", "ExecutionQualitySettings", "ExecutionRejected",
    "ExecutionResult", "ExecutionTelemetry", "ExecutionUncertain", "ExitSettings",
    "ForwardDemoQualifier", "ForwardQualificationReport", "ForwardQualificationSettings",
    "Intent", "IntentKind", "LabelOutcome", "LabelSettings", "LearningCycle",
    "LearningSettings", "MicroFeatures", "OnlineLogit", "PortfolioExposureEngine",
    "PortfolioSnapshot", "PositionState", "PromotionReport", "ProtectionSettings",
    "ReplayReport", "RiskController", "RuntimeSettings", "Sample", "ScalperCore",
    "ScalperRiskLimits", "ScalperStore", "SelfImprovementController", "Side",
    "StoredSample", "Tick", "TickFeatureEngine", "TickReplayBuilder",
]
