"""Core simulation engine for the mixed-autonomy ABM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.random import Generator

from src.agents import (
    AVAgent,
    HumanAgent,
    clip_aggression,
    create_av_population,
    create_human_population,
)


@dataclass
class SimulationConfig:
    n_agents: int = 200
    timesteps: int = 100
    av_prevalence: float = 0.25
    environment: str = "mixed"
    seed: int = 0
    av_aggression: float = 0.90
    human_influence_weight: float = 1.00
    base_seed: int = 42


@dataclass
class TimestepMetrics:
    timestep: int
    mean_human_aggression: float
    var_human_aggression: float
    mean_by_type: dict[str, float]
    mean_hh_encounter_aggression: float
    mean_ha_encounter_aggression: float
    n_hh_encounters: int
    n_ha_encounters: int


Agent = Union[HumanAgent, AVAgent]


class MixedAutonomyModel:
    """Agent-based model with dynamic random-mixing encounters."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng: Generator = np.random.default_rng(config.seed)

        n_av = round(config.n_agents * config.av_prevalence)
        n_human = config.n_agents - n_av

        self.humans = create_human_population(
            n_human, self.rng, config.environment, start_id=0
        )
        self.avs = create_av_population(
            n_av, aggression=config.av_aggression, start_id=n_human
        )
        self.agents: list[Agent] = list(self.humans) + list(self.avs)
        self._initial_av_aggressions = {av.agent_id: av.aggression for av in self.avs}
        self._initial_human_count = len(self.humans)
        self._initial_av_count = len(self.avs)
        self._current_timestep = 0

    def _pair_agents(self) -> list[tuple[Agent, Agent]]:
        """Shuffle agents and pair sequentially."""
        indices = self.rng.permutation(len(self.agents))
        pairs: list[tuple[Agent, Agent]] = []
        for i in range(0, len(indices) - 1, 2):
            pairs.append(
                (self.agents[indices[i]], self.agents[indices[i + 1]])
            )
        return pairs

    def _update_human_from_human(
        self, human: HumanAgent, observed_aggression: float
    ) -> float:
        difference = observed_aggression - human.baseline_aggression
        new_baseline = (
            human.baseline_aggression
            + human.susceptibility
            * self.config.human_influence_weight
            * difference
        )
        return clip_aggression(new_baseline)

    def _update_human_from_av(self, human: HumanAgent, av_aggression: float) -> float:
        difference = av_aggression - human.baseline_aggression
        new_baseline = (
            human.baseline_aggression
            + human.susceptibility * human.av_influence_weight * difference
        )
        return clip_aggression(new_baseline)

    def _compute_updates(
        self, pairs: list[tuple[Agent, Agent]]
    ) -> dict[int, float]:
        """Compute new baselines synchronously from pre-update values."""
        updates: dict[int, float] = {}

        for agent_a, agent_b in pairs:
            a_is_human = isinstance(agent_a, HumanAgent)
            b_is_human = isinstance(agent_b, HumanAgent)

            if a_is_human and b_is_human:
                human_a = agent_a
                human_b = agent_b
                old_a = human_a.baseline_aggression
                old_b = human_b.baseline_aggression

                diff_a = old_b - old_a
                new_a = clip_aggression(
                    old_a
                    + human_a.susceptibility
                    * self.config.human_influence_weight
                    * diff_a
                )
                diff_b = old_a - old_b
                new_b = clip_aggression(
                    old_b
                    + human_b.susceptibility
                    * self.config.human_influence_weight
                    * diff_b
                )
                updates[human_a.agent_id] = new_a
                updates[human_b.agent_id] = new_b

            elif a_is_human and not b_is_human:
                av_b = agent_b
                updates[agent_a.agent_id] = self._update_human_from_av(
                    agent_a, av_b.aggression
                )

            elif not a_is_human and b_is_human:
                av_a = agent_a
                updates[agent_b.agent_id] = self._update_human_from_av(
                    agent_b, av_a.aggression
                )

        return updates

    def _apply_updates(self, updates: dict[int, float]) -> None:
        human_by_id = {h.agent_id: h for h in self.humans}
        for agent_id, new_baseline in updates.items():
            human_by_id[agent_id].baseline_aggression = clip_aggression(new_baseline)

    def _record_metrics(
        self,
        timestep: int,
        hh_observations: list[float],
        ha_observations: list[float],
        n_hh_encounters: int,
        n_ha_encounters: int,
    ) -> TimestepMetrics:
        aggressions = [h.baseline_aggression for h in self.humans]
        mean_agg = float(np.mean(aggressions)) if aggressions else 0.0
        var_agg = float(np.var(aggressions)) if aggressions else 0.0

        mean_by_type: dict[str, float] = {}
        for response_type in ("assimilator", "discounter", "rejecter"):
            type_aggs = [
                h.baseline_aggression
                for h in self.humans
                if h.av_response_type.value == response_type
            ]
            mean_by_type[response_type] = (
                float(np.mean(type_aggs)) if type_aggs else float("nan")
            )

        mean_hh = float(np.mean(hh_observations)) if hh_observations else float("nan")
        mean_ha = float(np.mean(ha_observations)) if ha_observations else float("nan")

        return TimestepMetrics(
            timestep=timestep,
            mean_human_aggression=mean_agg,
            var_human_aggression=var_agg,
            mean_by_type=mean_by_type,
            mean_hh_encounter_aggression=mean_hh,
            mean_ha_encounter_aggression=mean_ha,
            n_hh_encounters=n_hh_encounters,
            n_ha_encounters=n_ha_encounters,
        )

    def _collect_encounter_observations(
        self, pairs: list[tuple[Agent, Agent]]
    ) -> tuple[list[float], list[float], int, int]:
        hh_observations: list[float] = []
        ha_observations: list[float] = []
        n_hh = 0
        n_ha = 0

        for agent_a, agent_b in pairs:
            a_is_human = isinstance(agent_a, HumanAgent)
            b_is_human = isinstance(agent_b, HumanAgent)

            if a_is_human and b_is_human:
                n_hh += 1
                hh_observations.append(agent_a.baseline_aggression)
                hh_observations.append(agent_b.baseline_aggression)
            elif a_is_human and not b_is_human:
                n_ha += 1
                ha_observations.append(agent_a.baseline_aggression)
                ha_observations.append(agent_b.aggression)
            elif not a_is_human and b_is_human:
                n_ha += 1
                ha_observations.append(agent_a.aggression)
                ha_observations.append(agent_b.baseline_aggression)

        return hh_observations, ha_observations, n_hh, n_ha

    def step(self) -> TimestepMetrics:
        """Run one timestep: pair, observe, update, record."""
        pairs = self._pair_agents()
        hh_obs, ha_obs, n_hh, n_ha = self._collect_encounter_observations(pairs)
        updates = self._compute_updates(pairs)
        self._apply_updates(updates)

        metrics = self._record_metrics(
            self._current_timestep, hh_obs, ha_obs, n_hh, n_ha
        )
        self._current_timestep += 1
        return metrics

    def record_initial_metrics(self) -> TimestepMetrics:
        """Record metrics at timestep 0 before any updates."""
        return self._record_metrics(0, [], [], 0, 0)

    def run(self) -> list[TimestepMetrics]:
        """Run full simulation and return trajectory including t=0."""
        trajectory = [self.record_initial_metrics()]
        for _ in range(self.config.timesteps):
            trajectory.append(self.step())
        return trajectory

    def get_final_stats(self) -> dict[str, float]:
        """Return summary statistics from current human population."""
        aggressions = [h.baseline_aggression for h in self.humans]
        stats: dict[str, float] = {
            "final_mean_aggression": float(np.mean(aggressions)) if aggressions else 0.0,
            "final_var_aggression": float(np.var(aggressions)) if aggressions else 0.0,
        }
        for response_type in ("assimilator", "discounter", "rejecter"):
            type_aggs = [
                h.baseline_aggression
                for h in self.humans
                if h.av_response_type.value == response_type
            ]
            stats[f"final_mean_{response_type}"] = (
                float(np.mean(type_aggs)) if type_aggs else float("nan")
            )
        return stats

    @property
    def initial_human_count(self) -> int:
        return self._initial_human_count

    @property
    def initial_av_count(self) -> int:
        return self._initial_av_count

    @property
    def initial_av_aggressions(self) -> dict[int, float]:
        return self._initial_av_aggressions
