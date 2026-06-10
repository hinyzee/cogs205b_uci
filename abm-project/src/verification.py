"""Runtime verification checks for simulation integrity."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.agents import HumanAgent, create_avs
from src.model import SimulationParams, run_single_simulation, run_timestep


class VerificationError(Exception):
    """Raised when one or more verification checks fail."""


def verify_bounds(trajectory_df: pd.DataFrame) -> tuple[bool, str]:
    """All recorded human aggression summaries must be within [0, 1]."""
    cols = [
        "mean_human_aggression",
        "mean_aggression_assimilator",
        "mean_aggression_discounter",
        "mean_aggression_rejecter",
    ]
    for col in cols:
        if col not in trajectory_df.columns:
            continue
        values = trajectory_df[col].dropna()
        if len(values) == 0:
            continue
        if values.min() < 0.0 or values.max() > 1.0:
            return False, f"Bounds check failed for column {col}"
    return True, "Bounds check passed"


def verify_population_constant(trajectory_df: pd.DataFrame) -> tuple[bool, str]:
    """Human and AV counts must remain constant within each run."""
    grouped = trajectory_df.groupby("run_id")
    for run_id, group in grouped:
        if group["n_humans"].nunique() != 1:
            return False, f"Population check failed: varying n_humans in {run_id}"
        if group["n_avs"].nunique() != 1:
            return False, f"Population check failed: varying n_avs in {run_id}"
    return True, "Population check passed"


def verify_seed_determinism(
    params: SimulationParams,
    run_fn: Callable[[SimulationParams], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> tuple[bool, str]:
    """Same seed must produce identical trajectories."""
    runner = run_fn or run_single_simulation
    traj_a, enc_a = runner(params)
    traj_b, enc_b = runner(params)
    if not traj_a.equals(traj_b) or not enc_a.equals(enc_b):
        return False, "Seed determinism check failed"
    return True, "Seed determinism check passed"


def verify_null_condition(
    trajectory_df: pd.DataFrame,
    tolerance: float = 0.02,
) -> tuple[bool, str]:
    """At 0% AV prevalence, environments should produce similar mean aggression."""
    null_df = trajectory_df[trajectory_df["av_prevalence"] == 0.0]
    if null_df.empty:
        return True, "Null check skipped: no 0% AV data"

    final = null_df[null_df["timestep"] == null_df["timestep"].max()]
    env_means = final.groupby("environment")["mean_human_aggression"].mean()
    if env_means.max() - env_means.min() > tolerance:
        return (
            False,
            "Null check failed: 0% AV environments diverged beyond tolerance",
        )
    return True, "Null check passed"


def verify_av_fixed(
    encounter_df: pd.DataFrame,
    expected: float = 0.90,
) -> tuple[bool, str]:
    """AV aggression in human-AV encounters must remain fixed."""
    ha_rows = encounter_df.dropna(subset=["mean_av_ha_encounter_aggression"])
    ha_rows = ha_rows[ha_rows["n_ha_encounters"] > 0]
    if ha_rows.empty:
        return True, "AV update check passed (no human-AV encounters)"

    if not np.allclose(
        ha_rows["mean_av_ha_encounter_aggression"].values,
        expected,
        atol=1e-9,
    ):
        return False, "AV update check failed: AV aggression changed"
    return True, "AV update check passed"


def verify_response_types(
    av_aggression: float = 0.90,
    human_influence_weight: float = 1.0,
) -> tuple[bool, str]:
    """Micro-simulation verifying assimilator, discounter, and rejecter responses."""
    av = create_avs(1, aggression=av_aggression)[0]
    rng = np.random.default_rng(0)

    assimilator = HumanAgent(0.40, 0.10, "assimilator", 0)
    discounter = HumanAgent(0.40, 0.10, "discounter", 1)
    rejecter = HumanAgent(0.40, 0.10, "rejecter", 2)

    for _ in range(30):
        run_timestep(
            [assimilator, av],
            rng,
            human_influence_weight=human_influence_weight,
        )

    discounter_before = discounter.baseline_aggression
    rejecter_before = rejecter.baseline_aggression

    for _ in range(30):
        run_timestep(
            [discounter, av],
            rng,
            human_influence_weight=human_influence_weight,
        )
        run_timestep(
            [rejecter, av],
            rng,
            human_influence_weight=human_influence_weight,
        )

    if assimilator.baseline_aggression <= 0.40:
        return False, "Response-type check failed: assimilator did not move toward AV"
    if discounter.baseline_aggression != discounter_before:
        return False, "Response-type check failed: discounter updated from AV"
    if rejecter.baseline_aggression >= rejecter_before:
        return False, "Response-type check failed: rejecter did not move away from AV"
    return True, "Response-type check passed"


def verify_multi_seed_aggregation(
    trajectory_df: pd.DataFrame,
    expected_seeds: int,
) -> tuple[bool, str]:
    """Each condition should include the expected number of seeds."""
    grouped = trajectory_df.groupby(["av_prevalence", "environment"])["seed"].nunique()
    if not (grouped == expected_seeds).all():
        return False, "Sensitivity check failed: unexpected seed counts per condition"
    return True, "Sensitivity check passed"


def verify_timesteps(
    trajectory_df: pd.DataFrame,
    timesteps: int,
) -> tuple[bool, str]:
    """Each run must have unique timesteps from 0 to timesteps inclusive."""
    for run_id, group in trajectory_df.groupby("run_id"):
        ts = group["timestep"].tolist()
        expected = list(range(timesteps + 1))
        if ts != expected:
            return False, f"Timestep check failed for {run_id}"
        if group["timestep"].duplicated().any():
            return False, f"Timestep check failed: duplicates in {run_id}"
    return True, "Timestep check passed"


def run_all_verifications(
    trajectory_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
    *,
    timesteps: int,
    expected_seeds: int,
    params: SimulationParams | None = None,
) -> None:
    """Run all verification checks and raise VerificationError on failure."""
    checks = [
        verify_bounds(trajectory_df),
        verify_population_constant(trajectory_df),
        verify_null_condition(trajectory_df),
        verify_av_fixed(encounter_df),
        verify_response_types(),
        verify_multi_seed_aggregation(trajectory_df, expected_seeds),
        verify_timesteps(trajectory_df, timesteps),
    ]

    if params is not None:
        checks.insert(2, verify_seed_determinism(params))

    failures = [message for passed, message in checks if not passed]
    if failures:
        detail = "\n".join(f"- {msg}" for msg in failures)
        raise VerificationError(f"Verification failed:\n{detail}")
