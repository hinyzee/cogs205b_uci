"""Tests for experiment grid and aggregation."""

from src.experiments import (
    DEFAULT_PARAMS,
    EXPERIMENT_GRID,
    ExperimentParams,
    aggregate_by_timestep,
    aggregate_final,
    condition_seed,
    run_all_conditions,
)


def test_grid_has_nine_conditions():
    assert len(EXPERIMENT_GRID) == 9


def test_hierarchical_seeds_stable():
    seed_a = condition_seed(2026, 3, 7)
    seed_b = condition_seed(2026, 3, 7)
    seed_c = condition_seed(2026, 4, 7)
    assert seed_a == seed_b
    assert seed_a != seed_c


def test_aggregation_has_thirty_seeds_per_condition():
    params = ExperimentParams(n_agents=20, timesteps=5, seeds=3, base_seed=1)
    trajectory, encounter = run_all_conditions(params)

    summary = aggregate_final(trajectory, encounter)
    assert len(summary) == 9
    assert (summary["n_seeds"] == 3).all()

    by_timestep = aggregate_by_timestep(trajectory, encounter)
    grouped = by_timestep.groupby(["av_prevalence", "environment"])["n_seeds"].max()
    assert (grouped == 3).all()
