"""Experiment grid runner and CSV output for the mixed-autonomy ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.agents import AV_PREVALENCE_LEVELS, ENVIRONMENT_IDS, ENVIRONMENTS, PREVALENCE_IDS
from src.model import MixedAutonomyModel, SimulationConfig, TimestepMetrics


@dataclass
class ExperimentGrid:
    n_agents: int = 200
    timesteps: int = 100
    n_seeds: int = 30
    base_seed: int = 42
    av_aggression: float = 0.90
    human_influence_weight: float = 1.00
    environments: list[str] | None = None
    av_prevalence_levels: list[float] | None = None

    def __post_init__(self) -> None:
        if self.environments is None:
            self.environments = list(ENVIRONMENTS.keys())
        if self.av_prevalence_levels is None:
            self.av_prevalence_levels = AV_PREVALENCE_LEVELS[:]

    def run_seed(self, environment: str, av_prevalence: float, seed_index: int) -> int:
        """Compute deterministic hierarchical seed for a run."""
        env_id = ENVIRONMENT_IDS[environment]
        prev_id = PREVALENCE_IDS[av_prevalence]
        return self.base_seed + env_id * 1000 + prev_id * 100 + seed_index

    def make_config(
        self, environment: str, av_prevalence: float, seed_index: int
    ) -> SimulationConfig:
        return SimulationConfig(
            n_agents=self.n_agents,
            timesteps=self.timesteps,
            av_prevalence=av_prevalence,
            environment=environment,
            seed=self.run_seed(environment, av_prevalence, seed_index),
            av_aggression=self.av_aggression,
            human_influence_weight=self.human_influence_weight,
            base_seed=self.base_seed,
        )


def metrics_to_row(
    metrics: TimestepMetrics,
    seed: int,
    environment: str,
    av_prevalence: float,
) -> dict:
    """Convert TimestepMetrics to a flat dict for DataFrame."""
    row = {
        "seed": seed,
        "environment": environment,
        "av_prevalence": av_prevalence,
        "timestep": metrics.timestep,
        "mean_human_aggression": metrics.mean_human_aggression,
        "var_human_aggression": metrics.var_human_aggression,
        "mean_assimilator": metrics.mean_by_type.get("assimilator", float("nan")),
        "mean_discounter": metrics.mean_by_type.get("discounter", float("nan")),
        "mean_rejecter": metrics.mean_by_type.get("rejecter", float("nan")),
        "mean_hh_encounter_aggression": metrics.mean_hh_encounter_aggression,
        "mean_ha_encounter_aggression": metrics.mean_ha_encounter_aggression,
        "n_hh_encounters": metrics.n_hh_encounters,
        "n_ha_encounters": metrics.n_ha_encounters,
    }
    return row


def run_single_simulation(
    config: SimulationConfig,
) -> tuple[pd.DataFrame, dict]:
    """Run one simulation and return trajectory DataFrame plus summary dict."""
    model = MixedAutonomyModel(config)
    trajectory = model.run()
    final_stats = model.get_final_stats()

    rows = [
        metrics_to_row(m, config.seed, config.environment, config.av_prevalence)
        for m in trajectory
    ]
    traj_df = pd.DataFrame(rows)

    summary = {
        "seed": config.seed,
        "environment": config.environment,
        "av_prevalence": config.av_prevalence,
        "initial_mean_aggression": trajectory[0].mean_human_aggression,
        **final_stats,
    }
    return traj_df, summary


def run_full_grid(grid: ExperimentGrid, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full factorial grid and write CSV outputs."""
    trajectories_dir = output_dir / "trajectories"
    summaries_dir = output_dir / "summaries"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    all_trajectories: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    for environment in grid.environments:
        for av_prevalence in grid.av_prevalence_levels:
            for seed_index in range(grid.n_seeds):
                config = grid.make_config(environment, av_prevalence, seed_index)
                traj_df, summary = run_single_simulation(config)
                all_trajectories.append(traj_df)
                all_summaries.append(summary)

                prev_label = int(av_prevalence * 100)
                traj_path = (
                    trajectories_dir
                    / f"{environment}_{prev_label}pct_seed{seed_index}.csv"
                )
                traj_df.to_csv(traj_path, index=False)

    run_summary_df = pd.DataFrame(all_summaries)
    run_summary_df.to_csv(summaries_dir / "run_summary.csv", index=False)

    full_trajectories = pd.concat(all_trajectories, ignore_index=True)
    aggregated = aggregate_across_seeds(full_trajectories, run_summary_df)
    aggregated["trajectories"].to_csv(
        summaries_dir / "aggregated_trajectories.csv", index=False
    )
    aggregated["summary"].to_csv(
        summaries_dir / "aggregated_summary.csv", index=False
    )

    return run_summary_df, aggregated["trajectories"]


def aggregate_across_seeds(
    trajectories_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Aggregate results across seeds by condition."""
    agg_traj = (
        trajectories_df.groupby(["environment", "av_prevalence", "timestep"])
        .agg(
            mean_human_aggression=("mean_human_aggression", "mean"),
            std_human_aggression=("mean_human_aggression", "std"),
            mean_var_aggression=("var_human_aggression", "mean"),
            std_var_aggression=("var_human_aggression", "std"),
            mean_hh_encounter_aggression=("mean_hh_encounter_aggression", "mean"),
            std_hh_encounter_aggression=("mean_hh_encounter_aggression", "std"),
        )
        .reset_index()
    )

    agg_summary = (
        summary_df.groupby(["environment", "av_prevalence"])
        .agg(
            mean_final_aggression=("final_mean_aggression", "mean"),
            std_final_aggression=("final_mean_aggression", "std"),
            mean_final_variance=("final_var_aggression", "mean"),
            std_final_variance=("final_var_aggression", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )

    return {"trajectories": agg_traj, "summary": agg_summary}
