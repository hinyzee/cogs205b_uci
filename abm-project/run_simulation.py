#!/usr/bin/env python3
"""Run the full mixed-autonomy ABM experiment grid and save outputs."""

from __future__ import annotations

import sys
from pathlib import Path

from src.experiments import (
    DEFAULT_PARAMS,
    aggregate_by_timestep,
    aggregate_final,
    run_all_conditions,
)
from src.model import SimulationParams
from src.plotting import generate_all_plots
from src.verification import VerificationError, run_all_verifications

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running full experiment grid...")
    trajectory_df, encounter_df = run_all_conditions(DEFAULT_PARAMS)

    print("Running verification checks...")
    sample_condition = SimulationParams(
        n_agents=DEFAULT_PARAMS.n_agents,
        timesteps=DEFAULT_PARAMS.timesteps,
        av_prevalence=0.25,
        environment="mixed",
        av_aggression=DEFAULT_PARAMS.av_aggression,
        human_influence_weight=DEFAULT_PARAMS.human_influence_weight,
        seed=DEFAULT_PARAMS.base_seed,
    )
    try:
        run_all_verifications(
            trajectory_df,
            encounter_df,
            timesteps=DEFAULT_PARAMS.timesteps,
            expected_seeds=DEFAULT_PARAMS.seeds,
            params=sample_condition,
        )
    except VerificationError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Aggregating results...")
    summary_by_timestep = aggregate_by_timestep(trajectory_df, encounter_df)
    summary_final = aggregate_final(trajectory_df, encounter_df)

    print("Saving CSV outputs...")
    trajectory_df.to_csv(RESULTS_DIR / "trajectories.csv", index=False)
    encounter_df.to_csv(RESULTS_DIR / "encounter_metrics.csv", index=False)
    summary_by_timestep.to_csv(RESULTS_DIR / "summary_by_timestep.csv", index=False)
    summary_final.to_csv(RESULTS_DIR / "summary_final.csv", index=False)

    print("Generating plots...")
    generate_all_plots(summary_by_timestep, summary_final, PLOTS_DIR)

    print("Simulation complete.")
    print(f"Results saved to: {RESULTS_DIR.resolve()}")
    print("Verification: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
