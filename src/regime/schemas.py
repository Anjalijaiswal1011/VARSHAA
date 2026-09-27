"""
Schema & Data Structures for Weather Regime Engine.
Defines the six canonical regime categories, transition states,
and structured output contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class WeatherRegime(str, Enum):
    ACTIVE_MONSOON = "ACTIVE_MONSOON"
    BREAK_MONSOON = "BREAK_MONSOON"
    MONSOON_DEPRESSION = "MONSOON_DEPRESSION"
    COASTAL = "COASTAL"
    OROGRAPHIC = "OROGRAPHIC"
    WESTERN_DISTURBANCE = "WESTERN_DISTURBANCE"
    REGIME_UNCERTAIN = "REGIME_UNCERTAIN"


REGIME_CLASSES: List[str] = [
    WeatherRegime.ACTIVE_MONSOON.value,
    WeatherRegime.BREAK_MONSOON.value,
    WeatherRegime.MONSOON_DEPRESSION.value,
    WeatherRegime.COASTAL.value,
    WeatherRegime.OROGRAPHIC.value,
    WeatherRegime.WESTERN_DISTURBANCE.value,
]

REGIME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(REGIME_CLASSES)}
IDX_TO_REGIME: Dict[int, str] = {i: name for i, name in enumerate(REGIME_CLASSES)}


class TransitionStatus(str, Enum):
    STABLE = "STABLE"
    TRANSITION = "TRANSITION"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class RegimeProbabilityVector:
    """Continuous probability distribution over the six weather regimes."""
    active_monsoon: float
    break_monsoon: float
    monsoon_depression: float
    coastal: float
    orographic: float
    western_disturbance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.active_monsoon,
                self.break_monsoon,
                self.monsoon_depression,
                self.coastal,
                self.orographic,
                self.western_disturbance,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> RegimeProbabilityVector:
        # Clip negative weights to 0.0, then normalize to ensure sum is exactly 1.0
        clipped = np.maximum(np.asarray(arr, dtype=np.float32), 0.0)
        total = float(np.sum(clipped))
        if total > 0:
            norm = (clipped / total).tolist()
        else:
            norm = [1.0 / len(REGIME_CLASSES)] * len(REGIME_CLASSES)

        return cls(
            active_monsoon=round(norm[0], 4),
            break_monsoon=round(norm[1], 4),
            monsoon_depression=round(norm[2], 4),
            coastal=round(norm[3], 4),
            orographic=round(norm[4], 4),
            western_disturbance=round(norm[5], 4),
        )

    def dominant_regime(self) -> Tuple[str, float]:
        arr = self.to_array()
        idx = int(np.argmax(arr))
        return REGIME_CLASSES[idx], float(arr[idx])


@dataclass
class RegimeTransitionOutput:
    """Full structured output of the regime intelligence and transition layer."""
    timestamp: str
    grid_id: str
    current_regime: str
    current_regime_probabilities: Dict[str, float]
    previous_regime: Optional[str]
    transition_status: TransitionStatus
    transition_from: Optional[str]
    transition_to: Optional[str]
    transition_strength: float
    transition_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["transition_status"] = self.transition_status.value
        return d
