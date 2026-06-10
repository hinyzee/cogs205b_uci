"""Simulation dynamics: pairing, updates, and trajectory recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.agents import (
    DEFAULT_AV_AGGRESSION,
    DEFAULT_HUMAN_INFLUENCE_WEIGHT,
    Agent,
    AVAgent,
    EnvironmentName,
    HumanAgent,
    build_population,
    clip_aggression,
)


@dataclass(frozen=True)
class SimulationParams:
    n_agents: int = 200
    timesteps: int = 100
    av_prevalence: float = 0.0
    environment: EnvironmentName = "mixed"
    av_aggression: float = DEFAULT_AV_AGGRESSION
    human_influence_weight: float = DEFAULT_HUMAN_INFLUENCE_WEIGHT
    seed: int = 0


@dataclass
class EncounterStats:
    human_hh_values: list[float]
    human_ha_values: list[float]
    av_ha_values: list[float]

    @property
    def n_hh_encounters(self) -> int:
        return len(self.human_hh_values) // 2 if self.human_hh_values else 0

    @property
    def n_ha_encounters(self) -> int:
        return len(self.human_ha_values)


def pair_agents(
    agents: list[Agent],
    rng: np.random.Generator,
) -> tuple[list[tuple[Agent, Agent]], Agent | None]:
    """Shuffle agents and pair them; leave one random agent unpaired if count is odd."""
    shuffled = list(agents)
    rng.shuffle(shuffled)

    skipped: Agent | None = None
    if len(shuffled) % 2 == 1:
        skip_idx = int(rng.integers(0, len(shuffled)))
        skipped = shuffled.pop(skip_idx)

    pairs = [
        (shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)
    ]
    return pairs, skipped


def compute_human_human_update(
    human_i: HumanAgent,
    human_j: HumanAgent,
    *,
    old_i: float,
    old_j: float,
    human_influence_weight: float,
) -> tuple[float, float]:
    """Compute simultaneous human-human update from pre-update baselines."""
    new_i = old_i + human_i.susceptibility * human_influence_weight * (old_j - old_i)
    new_j = old_j + human_j.susceptibility * human_influence_weight * (old_i - old_j)
    return clip_aggression(new_i), clip_aggression(new_j)


def compute_human_av_update(
    human: HumanAgent,
    av: AVAgent,
    *,
    old_baseline: float,
) -> float:
    """Compute human update after observing an AV."""
    delta = (
        human.susceptibility
        * human.av_influence_weight
        * (av.aggression - old_baseline)
    )
    return clip_aggression(old_baseline + delta)


def _snapshot_human_baselines(population: list[Agent]) -> dict[int, float]:
    return {
        agent.agent_id: agent.baseline_aggression
        for agent in population
        if isinstance(agent, HumanAgent)
    }


def _apply_encounter_updates(
    pair: tuple[Agent, Agent],
    baselines: dict[int, float],
    human_influence_weight: float,
) -> dict[int, float]:
    """Return updated human baselines for agents in this pair."""
    a, b = pair
    updates: dict[int, float] = {}

    if isinstance(a, HumanAgent) and isinstance(b, HumanAgent):
        old_a = baselines[a.agent_id]
        old_b = baselines[b.agent_id]
        new_a, new_b = compute_human_human_update(
            a,
            b,
            old_i=old_a,
            old_j=old_b,
            human_influence_weight=human_influence_weight,
        )
        updates[a.agent_id] = new_a
        updates[b.agent_id] = new_b
    elif isinstance(a, HumanAgent) and isinstance(b, AVAgent):
        updates[a.agent_id] = compute_human_av_update(
            a, b, old_baseline=baselines[a.agent_id]
        )
    elif isinstance(a, AVAgent) and isinstance(b, HumanAgent):
        updates[b.agent_id] = compute_human_av_update(
            b, a, old_baseline=baselines[b.agent_id]
        )

    return updates


def _collect_encounter_stats(
    pairs: list[tuple[Agent, Agent]],
    baselines: dict[int, float],
) -> EncounterStats:
    human_hh_values: list[float] = []
    human_ha_values: list[float] = []
    av_ha_values: list[float] = []

    for a, b in pairs:
        if isinstance(a, HumanAgent) and isinstance(b, HumanAgent):
            human_hh_values.append(baselines[a.agent_id])
            human_hh_values.append(baselines[b.agent_id])
        elif isinstance(a, HumanAgent) and isinstance(b, AVAgent):
            human_ha_values.append(baselines[a.agent_id])
            av_ha_values.append(b.aggression)
        elif isinstance(a, AVAgent) and isinstance(b, HumanAgent):
            human_ha_values.append(baselines[b.agent_id])
            av_ha_values.append(a.aggression)

    return EncounterStats(
        human_hh_values=human_hh_values,
        human_ha_values=human_ha_values,
        av_ha_values=av_ha_values,
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _var_or_nan(values: list[float]) -> float:
    return float(np.var(values)) if values else float("nan")


def _record_metrics(
    humans: list[HumanAgent],
    encounter_stats: EncounterStats,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggressions = [h.baseline_aggression for h in humans]

    by_type: dict[str, list[float]] = {
        "assimilator": [],
        "discounter": [],
        "rejecter": [],
    }
    for human in humans:
        by_type[human.av_response_type].append(human.baseline_aggression)

    trajectory_row = {
        **meta,
        "n_humans": len(humans),
        "mean_human_aggression": float(np.mean(aggressions)),
        "var_human_aggression": float(np.var(aggressions)),
        "mean_aggression_assimilator": _mean_or_nan(by_type["assimilator"]),
        "mean_aggression_discounter": _mean_or_nan(by_type["discounter"]),
        "mean_aggression_rejecter": _mean_or_nan(by_type["rejecter"]),
    }

    encounter_row = {
        **meta,
        "mean_human_hh_encounter_aggression": _mean_or_nan(
            encounter_stats.human_hh_values
        ),
        "mean_human_ha_encounter_aggression": _mean_or_nan(
            encounter_stats.human_ha_values
        ),
        "mean_av_ha_encounter_aggression": _mean_or_nan(encounter_stats.av_ha_values),
        "n_hh_encounters": encounter_stats.n_hh_encounters,
        "n_ha_encounters": encounter_stats.n_ha_encounters,
    }

    return trajectory_row, encounter_row


def run_timestep(
    population: list[Agent],
    rng: np.random.Generator,
    *,
    human_influence_weight: float,
) -> EncounterStats:
    """Apply one update round and mutate human baselines in place."""
    baselines = _snapshot_human_baselines(population)
    pairs, _skipped = pair_agents(population, rng)
    encounter_stats = _collect_encounter_stats(pairs, baselines)

    pending_updates: dict[int, float] = {}
    for pair in pairs:
        pair_updates = _apply_encounter_updates(
            pair, baselines, human_influence_weight
        )
        pending_updates.update(pair_updates)

    for agent in population:
        if isinstance(agent, HumanAgent) and agent.agent_id in pending_updates:
            agent.baseline_aggression = pending_updates[agent.agent_id]

    return encounter_stats


def _empty_encounter_stats() -> EncounterStats:
    return EncounterStats(human_hh_values=[], human_ha_values=[], av_ha_values=[])


def run_single_simulation(params: SimulationParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one simulation and return trajectory and encounter metric DataFrames."""
    rng = np.random.default_rng(params.seed)
    population, n_humans, n_avs = build_population(
        params.n_agents,
        params.av_prevalence,
        params.environment,
        rng,
        av_aggression=params.av_aggression,
    )

    humans = [agent for agent in population if isinstance(agent, HumanAgent)]
    run_id = (
        f"prev{params.av_prevalence:.2f}_"
        f"{params.environment}_seed{params.seed}"
    )

    trajectory_rows: list[dict[str, Any]] = []
    encounter_rows: list[dict[str, Any]] = []

    for timestep in range(params.timesteps + 1):
        meta = {
            "run_id": run_id,
            "seed": params.seed,
            "av_prevalence": params.av_prevalence,
            "environment": params.environment,
            "timestep": timestep,
            "n_avs": n_avs,
        }

        if timestep == 0:
            encounter_stats = _empty_encounter_stats()
        else:
            encounter_stats = run_timestep(
                population,
                rng,
                human_influence_weight=params.human_influence_weight,
            )

        trajectory_row, encounter_row = _record_metrics(
            humans, encounter_stats, meta
        )
        trajectory_rows.append(trajectory_row)
        encounter_rows.append(encounter_row)

    return pd.DataFrame(trajectory_rows), pd.DataFrame(encounter_rows)
