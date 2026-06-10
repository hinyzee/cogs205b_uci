"""Tests for simulation dynamics and timestep conventions."""

import numpy as np
import pandas as pd
import pytest

from src.agents import AVAgent, HumanAgent, create_avs
from src.model import (
    SimulationParams,
    compute_human_av_update,
    compute_human_human_update,
    run_single_simulation,
    run_timestep,
)


def _make_human(
    aggression: float,
    susceptibility: float,
    response_type: str,
    agent_id: int,
) -> HumanAgent:
    return HumanAgent(
        baseline_aggression=aggression,
        susceptibility=susceptibility,
        av_response_type=response_type,
        agent_id=agent_id,
    )


def test_human_human_uses_pre_update_values():
    human_i = _make_human(0.40, 0.10, "assimilator", 0)
    human_j = _make_human(0.60, 0.10, "assimilator", 1)

    new_i, new_j = compute_human_human_update(
        human_i,
        human_j,
        old_i=0.40,
        old_j=0.60,
        human_influence_weight=1.0,
    )

    assert new_i == pytest.approx(0.42)
    assert new_j == pytest.approx(0.58)


def test_human_av_updates_human_only():
    human = _make_human(0.40, 0.10, "assimilator", 0)
    av = AVAgent(aggression=0.90, agent_id=1)

    new_baseline = compute_human_av_update(human, av, old_baseline=0.40)
    assert new_baseline == pytest.approx(0.425)
    assert av.aggression == 0.90


def test_av_never_updates():
    params = SimulationParams(
        n_agents=20,
        timesteps=20,
        av_prevalence=0.5,
        environment="mixed",
        seed=123,
    )
    rng = np.random.default_rng(params.seed)
    avs = create_avs(10, aggression=0.90)
    population = avs + [
        _make_human(0.35, 0.05, "assimilator", idx + 10) for idx in range(10)
    ]
    rng.shuffle(population)

    for _ in range(params.timesteps):
        run_timestep(population, rng, human_influence_weight=1.0)

    for agent in population:
        if isinstance(agent, AVAgent):
            assert agent.aggression == 0.90


def test_timestep_indexing():
    params = SimulationParams(n_agents=20, timesteps=5, seed=1)
    trajectory, _ = run_single_simulation(params)
    assert trajectory["timestep"].tolist() == [0, 1, 2, 3, 4, 5]


def test_no_duplicate_timesteps():
    params = SimulationParams(n_agents=20, timesteps=5, seed=2)
    trajectory, _ = run_single_simulation(params)
    assert trajectory["timestep"].is_unique


def test_trajectory_length_is_T_plus_1():
    timesteps = 7
    params = SimulationParams(n_agents=20, timesteps=timesteps, seed=3)
    trajectory, encounter = run_single_simulation(params)
    assert len(trajectory) == timesteps + 1
    assert len(encounter) == timesteps + 1


def test_same_seed_identical_trajectory():
    params_a = SimulationParams(n_agents=30, timesteps=10, av_prevalence=0.25, seed=99)
    params_b = SimulationParams(
        n_agents=30,
        timesteps=10,
        av_prevalence=0.25,
        environment=params_a.environment,
        seed=99,
    )
    traj_a, enc_a = run_single_simulation(params_a)
    traj_b, enc_b = run_single_simulation(params_b)
    pd.testing.assert_frame_equal(traj_a, traj_b)
    pd.testing.assert_frame_equal(enc_a, enc_b)


def test_encounter_metrics_separated_at_timestep_zero():
    params = SimulationParams(n_agents=20, timesteps=3, av_prevalence=0.5, seed=4)
    _, encounter = run_single_simulation(params)
    row0 = encounter.iloc[0]
    assert np.isnan(row0["mean_human_hh_encounter_aggression"])
    assert np.isnan(row0["mean_human_ha_encounter_aggression"])
    assert row0["n_hh_encounters"] == 0
    assert row0["n_ha_encounters"] == 0
