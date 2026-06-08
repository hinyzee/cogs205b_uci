"""Agent types and population factories for the mixed-autonomy ABM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from numpy.random import Generator


class AvResponseType(Enum):
    ASSIMILATOR = "assimilator"
    DISCOUNTER = "discounter"
    REJECTER = "rejecter"


AV_INFLUENCE_WEIGHTS: dict[AvResponseType, float] = {
    AvResponseType.ASSIMILATOR: 0.50,
    AvResponseType.DISCOUNTER: 0.00,
    AvResponseType.REJECTER: -0.50,
}

ENVIRONMENTS: dict[str, dict[str, float]] = {
    "mostly_assimilation": {"assimilator": 0.70, "discounter": 0.20, "rejecter": 0.10},
    "mixed": {"assimilator": 0.45, "discounter": 0.10, "rejecter": 0.45},
    "mostly_rejection": {"assimilator": 0.10, "discounter": 0.20, "rejecter": 0.70},
}

AV_PREVALENCE_LEVELS: list[float] = [0.00, 0.25, 0.50]

ENVIRONMENT_IDS: dict[str, int] = {
    "mostly_assimilation": 0,
    "mixed": 1,
    "mostly_rejection": 2,
}

PREVALENCE_IDS: dict[float, int] = {0.00: 0, 0.25: 1, 0.50: 2}


def clip_aggression(value: float) -> float:
    """Clip aggression to [0, 1]."""
    return min(1.0, max(0.0, value))


@dataclass
class HumanAgent:
    agent_id: int
    baseline_aggression: float
    susceptibility: float
    av_response_type: AvResponseType
    av_influence_weight: float

    def observed_aggression(self) -> float:
        return self.baseline_aggression


@dataclass
class AVAgent:
    agent_id: int
    aggression: float

    def observed_aggression(self) -> float:
        return self.aggression


def assign_response_types(
    n_humans: int,
    proportions: dict[str, float],
    rng: Generator,
) -> list[AvResponseType]:
    """Assign response types using largest-remainder method, then shuffle."""
    type_order = [
        AvResponseType.ASSIMILATOR,
        AvResponseType.DISCOUNTER,
        AvResponseType.REJECTER,
    ]
    keys = ["assimilator", "discounter", "rejecter"]

    raw_counts = [proportions[k] * n_humans for k in keys]
    floors = [int(c) for c in raw_counts]
    remainders = [raw - fl for raw, fl in zip(raw_counts, floors)]
    deficit = n_humans - sum(floors)

    counts = floors[:]
    for idx in np.argsort(remainders)[::-1][:deficit]:
        counts[idx] += 1

    assignments: list[AvResponseType] = []
    for response_type, count in zip(type_order, counts):
        assignments.extend([response_type] * count)

    rng.shuffle(assignments)
    return assignments


def create_human_population(
    n: int,
    rng: Generator,
    environment: str,
    start_id: int = 0,
) -> list[HumanAgent]:
    """Create human agents with randomized aggression and susceptibility."""
    proportions = ENVIRONMENTS[environment]
    response_types = assign_response_types(n, proportions, rng)

    aggressions = rng.normal(0.35, 0.08, size=n)
    aggressions = np.clip(aggressions, 0.0, 1.0)
    susceptibilities = rng.uniform(0.02, 0.12, size=n)

    humans: list[HumanAgent] = []
    for i in range(n):
        response_type = response_types[i]
        humans.append(
            HumanAgent(
                agent_id=start_id + i,
                baseline_aggression=float(aggressions[i]),
                susceptibility=float(susceptibilities[i]),
                av_response_type=response_type,
                av_influence_weight=AV_INFLUENCE_WEIGHTS[response_type],
            )
        )
    return humans


def create_av_population(
    n: int,
    aggression: float = 0.90,
    start_id: int = 0,
) -> list[AVAgent]:
    """Create fixed-aggression AV agents."""
    return [
        AVAgent(agent_id=start_id + i, aggression=aggression) for i in range(n)
    ]
