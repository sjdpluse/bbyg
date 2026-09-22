"""BBYG tick-level scalping research and demo execution core."""

from .engine import EntrySettings, ScalperCore
from .features import TickFeatureEngine
from .learning import ChampionChallenger, OnlineLogit, PromotionReport, Sample
from .position import AlgorithmicExitManager, ExitSettings
from .risk import RiskController, ScalperRiskLimits
from .types import Intent, IntentKind, MicroFeatures, PositionState, Side, Tick

__all__ = [
    "AlgorithmicExitManager", "ChampionChallenger", "EntrySettings", "ExitSettings",
    "Intent", "IntentKind", "MicroFeatures", "OnlineLogit", "PositionState",
    "PromotionReport", "RiskController", "Sample", "ScalperCore", "ScalperRiskLimits",
    "Side", "Tick", "TickFeatureEngine",
]
