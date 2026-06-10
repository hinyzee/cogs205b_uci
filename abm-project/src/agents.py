"""Agent types, initialization, and population construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AVResponseType = Literal["assimilator", "discounter", "rejecter"]
EnvironmentName = Literal["mostly_assimilation", "mixed", "mostly_rejection"]

AV_INFLUENCE_WEIGHTS: dict[AVResponseType, float] = {
    "assimilator": 0.50,
    "discounter": 0.00,
    "rejecter": -0.50,
}

ENVIRONMENT_PROPORTIONS: dict[EnvironmentName, dict[AVResponseType, float]] = {
    "mostly_assimilation": {
        "assimilator": 0.70,
        "discounter": 0.20,
        "rejecter": 0.10,
    },
    "mixed": {
        "assimilator": 0.45,
        "discounter": 0.10,
        "rejecter": 0.45,
    },
    "mostly_rejection": {
        "assimilator": 0.10,
        "discounter": 0.20,
        "rejecter": 0.70,
    },
}

DEFAULT_AV_AGGRESSION = 0.90
DEFAULT_HUMAN_INFLUENCE_WEIGHT = 1.00
DEFAULT_INITIAL_AGGRESSION_MEAN = 0.35
DEFAULT_INITIAL_AGGRESSION_STD = 0.08
DEFAULT_SUSCEPTIBILITY_LOW = 0.02
DEFAULT_SUSCEPTIBILITY_HIGH = 0.12


def clip_aggression(value: float) -> float:
    """Clip human baseline aggression to [0, 1]."""
    return float(min(1.0, max(0.0, value)))


@dataclass
class HumanAgent:
    baseline_aggression: float
    susceptibility: float
    av_response_type: AVResponseType
    agent_id: int

    @property
    def av_influence_weight(self) -> float:
        return AV_INFLUENCE_WEIGHTS[self.av_response_type]

    @property
    def is_av(self) -> bool:
        return False


@dataclass
class AVAgent:
    aggression: float
    agent_id: int

    @property
    def is_av(self) -> bool:
        return True


Agent = HumanAgent | AVAgent


def assign_response_types(
    n: int,
    environment: EnvironmentName,
    rng: np.random.Generator,
) -> list[AVResponseType]:
    """Assign AV response types to n humans according to environment proportions."""
    proportions = ENVIRONMENT_PROPORTIONS[environment]
    types: list[AVResponseType] = ["assimilator", "discounter", "rejecter"]
    counts = rng.multinomial(n, [proportions[t] for t in types])
    assigned: list[AVResponseType] = []
    for response_type, count in zip(types, counts):
        assigned.extend([response_type] * count)
    rng.shuffle(assigned)
    return assigned


def create_humans(
    n: int,
    environment: EnvironmentName,
    rng: np.random.Generator,
    *,
    aggression_mean: float = DEFAULT_INITIAL_AGGRESSION_MEAN,
    aggression_std: float = DEFAULT_INITIAL_AGGRESSION_STD,
    susceptibility_low: float = DEFAULT_SUSCEPTIBILITY_LOW,
    susceptibility_high: float = DEFAULT_SUSCEPTIBILITY_HIGH,
    start_id: int = 0,
) -> list[HumanAgent]:
    """Create human agents with initialized aggression and susceptibility."""
    aggressions = rng.normal(aggression_mean, aggression_std, size=n)
    aggressions = np.clip(aggressions, 0.0, 1.0)
    susceptibilities = rng.uniform(susceptibility_low, susceptibility_high, size=n)
    response_types = assign_response_types(n, environment, rng)

    humans: list[HumanAgent] = []
    for idx in range(n):
        humans.append(
            HumanAgent(
                baseline_aggression=float(aggressions[idx]),
                susceptibility=float(susceptibilities[idx]),
                av_response_type=response_types[idx],
                agent_id=start_id + idx,
            )
        )
    return humans


def create_avs(
    n: int,
    aggression: float = DEFAULT_AV_AGGRESSION,
    start_id: int = 0,
) -> list[AVAgent]:
    """Create fixed-aggression AV agents."""
    return [
        AVAgent(aggression=aggression, agent_id=start_id + idx) for idx in range(n)
    ]


def build_population(
    n_agents: int,
    av_prevalence: float,
    environment: EnvironmentName,
    rng: np.random.Generator,
    *,
    av_aggression: float = DEFAULT_AV_AGGRESSION,
) -> tuple[list[Agent], int, int]:
    """Build a shuffled mixed population of humans and AVs."""
    n_avs = int(n_agents * av_prevalence)
    n_humans = n_agents - n_avs

    humans = create_humans(n_humans, environment, rng, start_id=0)
    avs = create_avs(n_avs, aggression=av_aggression, start_id=n_humans)

    population: list[Agent] = humans + avs
    rng.shuffle(population)
    return population, n_humans, n_avs
