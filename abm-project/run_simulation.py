#!/usr/bin/env python3
"""Entry point for the mixed-autonomy traffic ABM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiments import ExperimentGrid, run_full_grid
from src.model import SimulationConfig
from src.plotting import generate_all_plots
from src.verification import run_all_checks, run_simulation_checks, print_check_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mixed-autonomy traffic ABM simulations"
    )
    parser.add_argument(
        "--seeds", type=int, default=30, help="Number of random seeds per condition"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="Number of simulation timesteps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for CSV outputs and plots",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Base seed for hierarchical seeding"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    (output_dir / "summaries").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    smoke_config = SimulationConfig(
        n_agents=200,
        timesteps=args.timesteps,
        av_prevalence=0.25,
        environment="mixed",
        seed=args.base_seed,
        base_seed=args.base_seed,
    )

    print("Running pre-grid verification checks...")
    pre_results = run_simulation_checks(smoke_config)
    pre_passed = print_check_results(pre_results)
    if not pre_passed:
        print("\nPre-grid verification FAILED. Aborting.")
        return 1

    print(f"\nRunning full experiment grid ({args.seeds} seeds per condition)...")
    grid = ExperimentGrid(
        n_agents=200,
        timesteps=args.timesteps,
        n_seeds=args.seeds,
        base_seed=args.base_seed,
    )
    run_summary_df, _ = run_full_grid(grid, output_dir)

    print("\nRunning post-grid verification checks...")
    post_results = run_all_checks(
        smoke_config,
        summary_df=run_summary_df,
        expected_seeds=args.seeds,
    )
    all_passed = print_check_results(post_results)
    if not all_passed:
        print("\nPost-grid verification FAILED.")
        return 1

    print("\nGenerating plots...")
    generate_all_plots(output_dir)

    print("\nAggregated final mean aggression by condition:")
    agg_summary = run_summary_df.groupby(["environment", "av_prevalence"])[
        "final_mean_aggression"
    ].agg(["mean", "std"])
    print(agg_summary.to_string())

    print(f"\nResults saved to {output_dir.resolve()}")
    print("All verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
