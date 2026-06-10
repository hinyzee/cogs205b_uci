"""Tests for runtime verification helpers."""

import pandas as pd
import pytest

from src.model import SimulationParams, run_single_simulation
from src.verification import (
    verify_bounds,
    verify_response_types,
    verify_timesteps,
)


def test_bounds_check_passes_valid():
    params = SimulationParams(n_agents=20, timesteps=5, seed=10)
    trajectory, _ = run_single_simulation(params)
    passed, _ = verify_bounds(trajectory)
    assert passed


def test_bounds_check_fails_invalid():
    df = pd.DataFrame(
        {
            "mean_human_aggression": [0.5, 1.5],
            "mean_aggression_assimilator": [0.4, 0.5],
            "mean_aggression_discounter": [0.4, 0.5],
            "mean_aggression_rejecter": [0.4, 0.5],
        }
    )
    passed, message = verify_bounds(df)
    assert not passed
    assert "Bounds check failed" in message


def test_response_type_micro_simulation_passes():
    passed, message = verify_response_types()
    assert passed, message


def test_timestep_check_catches_duplicates():
    df = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_a"],
            "timestep": [0, 0, 1],
        }
    )
    passed, message = verify_timesteps(df, timesteps=1)
    assert not passed
    assert "Timestep check failed" in message
