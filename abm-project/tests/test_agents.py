"""Tests for agent initialization and helper functions."""

import numpy as np

from src.agents import (
    AV_INFLUENCE_WEIGHTS,
    ENVIRONMENT_PROPORTIONS,
    HumanAgent,
    assign_response_types,
    build_population,
    clip_aggression,
    create_avs,
    create_humans,
)


def test_clip_aggression_below_zero():
    assert clip_aggression(-0.5) == 0.0


def test_clip_aggression_above_one():
    assert clip_aggression(1.5) == 1.0


def test_clip_aggression_in_range():
    assert clip_aggression(0.42) == 0.42


def test_response_type_counts_match_environment():
    rng = np.random.default_rng(42)
    n = 1000
    for environment, proportions in ENVIRONMENT_PROPORTIONS.items():
        assigned = assign_response_types(n, environment, rng)
        counts = {
            "assimilator": assigned.count("assimilator"),
            "discounter": assigned.count("discounter"),
            "rejecter": assigned.count("rejecter"),
        }
        for response_type, expected_prop in proportions.items():
            observed_prop = counts[response_type] / n
            assert abs(observed_prop - expected_prop) < 0.05


def test_av_influence_weights():
    assert AV_INFLUENCE_WEIGHTS["assimilator"] == 0.50
    assert AV_INFLUENCE_WEIGHTS["discounter"] == 0.00
    assert AV_INFLUENCE_WEIGHTS["rejecter"] == -0.50

    human = HumanAgent(
        baseline_aggression=0.35,
        susceptibility=0.05,
        av_response_type="assimilator",
        agent_id=0,
    )
    assert human.av_influence_weight == 0.50


def test_av_fixed_aggression_initialization():
    avs = create_avs(5, aggression=0.90)
    assert len(avs) == 5
    assert all(av.aggression == 0.90 for av in avs)
    assert all(av.is_av for av in avs)


def test_create_humans_within_bounds():
    rng = np.random.default_rng(7)
    humans = create_humans(50, "mixed", rng)
    assert len(humans) == 50
    for human in humans:
        assert 0.0 <= human.baseline_aggression <= 1.0
        assert 0.02 <= human.susceptibility <= 0.12


def test_build_population_counts():
    rng = np.random.default_rng(99)
    population, n_humans, n_avs = build_population(200, 0.25, "mixed", rng)
    assert len(population) == 200
    assert n_humans == 150
    assert n_avs == 50
