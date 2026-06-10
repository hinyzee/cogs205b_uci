"""Experiment grid, hierarchical seeds, and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.agents import EnvironmentName
from src.model import SimulationParams, run_single_simulation


@dataclass(frozen=True)
class ExperimentParams:
    n_agents: int = 200
    timesteps: int = 100
    seeds: int = 30
    av_aggression: float = 0.90
    human_influence_weight: float = 1.00
    base_seed: int = 2026


DEFAULT_PARAMS = ExperimentParams()

AV_PREVALENCE_LEVELS = [0.00, 0.25, 0.50]
ENVIRONMENTS: list[EnvironmentName] = [
    "mostly_assimilation",
    "mixed",
    "mostly_rejection",
]

EXPERIMENT_GRID = list(product(AV_PREVALENCE_LEVELS, ENVIRONMENTS))


def condition_seed(base_seed: int, condition_idx: int, seed_idx: int) -> int:
    """Deterministic hierarchical seed for a condition and replicate."""
    return base_seed + condition_idx * 10_000 + seed_idx


def run_all_conditions(
    params: ExperimentParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full 3x3 experimental grid across all seeds."""
    params = params or DEFAULT_PARAMS

    trajectory_frames: list[pd.DataFrame] = []
    encounter_frames: list[pd.DataFrame] = []

    for condition_idx, (av_prevalence, environment) in enumerate(EXPERIMENT_GRID):
        for seed_idx in range(params.seeds):
            seed = condition_seed(params.base_seed, condition_idx, seed_idx)
            sim_params = SimulationParams(
                n_agents=params.n_agents,
                timesteps=params.timesteps,
                av_prevalence=av_prevalence,
                environment=environment,
                av_aggression=params.av_aggression,
                human_influence_weight=params.human_influence_weight,
                seed=seed,
            )
            trajectory, encounter = run_single_simulation(sim_params)
            trajectory_frames.append(trajectory)
            encounter_frames.append(encounter)

    return (
        pd.concat(trajectory_frames, ignore_index=True),
        pd.concat(encounter_frames, ignore_index=True),
    )


def aggregate_by_timestep(
    trajectory_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-timestep means and standard deviations across seeds."""
    traj_group_cols = ["av_prevalence", "environment", "timestep"]
    traj_agg = (
        trajectory_df.groupby(traj_group_cols)
        .agg(
            mean_human_aggression_mean=("mean_human_aggression", "mean"),
            mean_human_aggression_std=("mean_human_aggression", "std"),
            var_human_aggression_mean=("var_human_aggression", "mean"),
            var_human_aggression_std=("var_human_aggression", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    enc_group_cols = ["av_prevalence", "environment", "timestep"]
    enc_agg = (
        encounter_df.groupby(enc_group_cols)
        .agg(
            mean_human_hh_encounter_aggression_mean=(
                "mean_human_hh_encounter_aggression",
                "mean",
            ),
            mean_human_hh_encounter_aggression_std=(
                "mean_human_hh_encounter_aggression",
                "std",
            ),
            mean_human_ha_encounter_aggression_mean=(
                "mean_human_ha_encounter_aggression",
                "mean",
            ),
            mean_human_ha_encounter_aggression_std=(
                "mean_human_ha_encounter_aggression",
                "std",
            ),
        )
        .reset_index()
    )

    return traj_agg.merge(enc_agg, on=enc_group_cols, how="left")


def aggregate_final(
    trajectory_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate final-timestep outcomes across seeds for each condition."""
    max_timestep = trajectory_df["timestep"].max()
    final_traj = trajectory_df[trajectory_df["timestep"] == max_timestep]
    final_enc = encounter_df[encounter_df["timestep"] == max_timestep]

    traj_summary = (
        final_traj.groupby(["av_prevalence", "environment"])
        .agg(
            final_mean_human_aggression=("mean_human_aggression", "mean"),
            final_std_human_aggression=("mean_human_aggression", "std"),
            final_mean_variance=("var_human_aggression", "mean"),
            final_std_variance=("var_human_aggression", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    enc_summary = (
        final_enc.groupby(["av_prevalence", "environment"])
        .agg(
            final_mean_human_hh_encounter_aggression=(
                "mean_human_hh_encounter_aggression",
                "mean",
            ),
            final_mean_human_ha_encounter_aggression=(
                "mean_human_ha_encounter_aggression",
                "mean",
            ),
        )
        .reset_index()
    )

    return traj_summary.merge(
        enc_summary,
        on=["av_prevalence", "environment"],
        how="left",
    )
