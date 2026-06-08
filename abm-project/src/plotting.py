"""Plotting utilities for the mixed-autonomy ABM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}

ENV_COLORS = {
    "mostly_assimilation": COLORS["blue"],
    "mixed": COLORS["orange"],
    "mostly_rejection": COLORS["green"],
}

ENV_STYLES = {
    "mostly_assimilation": "-",
    "mixed": "--",
    "mostly_rejection": ":",
}

ENV_LABELS = {
    "mostly_assimilation": "Mostly assimilation",
    "mixed": "Mixed",
    "mostly_rejection": "Mostly rejection",
}

PREV_COLORS = {
    0.0: COLORS["gray"],
    0.25: COLORS["blue"],
    0.50: COLORS["red"],
}

PREV_LABELS = {
    0.0: "0% AVs",
    0.25: "25% AVs",
    0.50: "50% AVs",
}


def setup_plot_style() -> None:
    """Apply consistent matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_mean_aggression_over_time(agg_df: pd.DataFrame, outpath: Path) -> None:
    """Plot mean human baseline aggression over time by condition."""
    setup_plot_style()
    environments = agg_df["environment"].unique()
    fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 4), sharey=True)
    if len(environments) == 1:
        axes = [axes]

    for ax, env in zip(axes, environments):
        env_data = agg_df[agg_df["environment"] == env]
        for prev in sorted(env_data["av_prevalence"].unique()):
            prev_data = env_data[env_data["av_prevalence"] == prev].sort_values("timestep")
            ax.plot(
                prev_data["timestep"],
                prev_data["mean_human_aggression"],
                color=PREV_COLORS.get(prev, COLORS["black"]),
                linestyle=ENV_STYLES.get(env, "-"),
                label=PREV_LABELS.get(prev, f"{prev:.0%} AVs"),
            )
            if "std_human_aggression" in prev_data.columns:
                ax.fill_between(
                    prev_data["timestep"],
                    prev_data["mean_human_aggression"] - prev_data["std_human_aggression"],
                    prev_data["mean_human_aggression"] + prev_data["std_human_aggression"],
                    color=PREV_COLORS.get(prev, COLORS["black"]),
                    alpha=0.15,
                )
        ax.set_title(ENV_LABELS.get(env, env))
        ax.set_xlabel("Timestep")
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("Mean human baseline aggression")
    fig.suptitle("Mean aggression over time", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_variance_over_time(agg_df: pd.DataFrame, outpath: Path) -> None:
    """Plot variance in human baseline aggression over time."""
    setup_plot_style()
    environments = agg_df["environment"].unique()
    fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 4), sharey=True)
    if len(environments) == 1:
        axes = [axes]

    for ax, env in zip(axes, environments):
        env_data = agg_df[agg_df["environment"] == env]
        for prev in sorted(env_data["av_prevalence"].unique()):
            prev_data = env_data[env_data["av_prevalence"] == prev].sort_values("timestep")
            ax.plot(
                prev_data["timestep"],
                prev_data["mean_var_aggression"],
                color=PREV_COLORS.get(prev, COLORS["black"]),
                linestyle=ENV_STYLES.get(env, "-"),
                label=PREV_LABELS.get(prev, f"{prev:.0%} AVs"),
            )
            if "std_var_aggression" in prev_data.columns:
                ax.fill_between(
                    prev_data["timestep"],
                    prev_data["mean_var_aggression"] - prev_data["std_var_aggression"],
                    prev_data["mean_var_aggression"] + prev_data["std_var_aggression"],
                    color=PREV_COLORS.get(prev, COLORS["black"]),
                    alpha=0.15,
                )
        ax.set_title(ENV_LABELS.get(env, env))
        ax.set_xlabel("Timestep")
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("Variance in human baseline aggression")
    fig.suptitle("Variance in aggression over time", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_final_mean_by_condition(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Grouped bar chart of final mean aggression by condition."""
    setup_plot_style()
    environments = list(summary_df["environment"].unique())
    prevalences = sorted(summary_df["av_prevalence"].unique())
    x = range(len(prevalences))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, env in enumerate(environments):
        env_data = summary_df[summary_df["environment"] == env]
        means = [
            env_data[env_data["av_prevalence"] == p]["mean_final_aggression"].values[0]
            for p in prevalences
        ]
        offset = (i - len(environments) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            means,
            width,
            label=ENV_LABELS.get(env, env),
            color=ENV_COLORS.get(env, COLORS["black"]),
        )

    ax.set_xlabel("AV prevalence")
    ax.set_ylabel("Final mean human baseline aggression")
    ax.set_title("Final aggression by AV prevalence")
    ax.set_xticks(list(x))
    ax.set_xticklabels([PREV_LABELS.get(p, f"{p:.0%}") for p in prevalences])
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_final_variance_by_condition(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Grouped bar chart of final variance by condition."""
    setup_plot_style()
    environments = list(summary_df["environment"].unique())
    prevalences = sorted(summary_df["av_prevalence"].unique())
    x = range(len(prevalences))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, env in enumerate(environments):
        env_data = summary_df[summary_df["environment"] == env]
        means = [
            env_data[env_data["av_prevalence"] == p]["mean_final_variance"].values[0]
            for p in prevalences
        ]
        offset = (i - len(environments) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            means,
            width,
            label=ENV_LABELS.get(env, env),
            color=ENV_COLORS.get(env, COLORS["black"]),
        )

    ax.set_xlabel("AV prevalence")
    ax.set_ylabel("Final variance in human baseline aggression")
    ax.set_title("Polarization under mixed AV responses")
    ax.set_xticks(list(x))
    ax.set_xticklabels([PREV_LABELS.get(p, f"{p:.0%}") for p in prevalences])
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_hh_encounter_aggression_over_time(agg_df: pd.DataFrame, outpath: Path) -> None:
    """Plot mean aggression during human-human encounters over time."""
    setup_plot_style()
    environments = agg_df["environment"].unique()
    fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 4), sharey=True)
    if len(environments) == 1:
        axes = [axes]

    for ax, env in zip(axes, environments):
        env_data = agg_df[agg_df["environment"] == env]
        for prev in sorted(env_data["av_prevalence"].unique()):
            prev_data = env_data[env_data["av_prevalence"] == prev].sort_values("timestep")
            ax.plot(
                prev_data["timestep"],
                prev_data["mean_hh_encounter_aggression"],
                color=PREV_COLORS.get(prev, COLORS["black"]),
                linestyle=ENV_STYLES.get(env, "-"),
                label=PREV_LABELS.get(prev, f"{prev:.0%} AVs"),
            )
        ax.set_title(ENV_LABELS.get(env, env))
        ax.set_xlabel("Timestep")
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("Mean aggression in human-human encounters")
    fig.suptitle("Human-human encounter aggression over time", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results_dir: Path) -> None:
    """Generate all required and optional plots from aggregated CSVs."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg_traj_path = results_dir / "summaries" / "aggregated_trajectories.csv"
    agg_summary_path = results_dir / "summaries" / "aggregated_summary.csv"

    agg_traj_df = pd.read_csv(agg_traj_path)
    agg_summary_df = pd.read_csv(agg_summary_path)

    plot_mean_aggression_over_time(agg_traj_df, plots_dir / "mean_aggression_over_time.png")
    plot_variance_over_time(agg_traj_df, plots_dir / "variance_over_time.png")
    plot_final_mean_by_condition(agg_summary_df, plots_dir / "final_mean_by_condition.png")
    plot_final_variance_by_condition(agg_summary_df, plots_dir / "final_variance_by_condition.png")
    plot_hh_encounter_aggression_over_time(
        agg_traj_df, plots_dir / "hh_encounter_aggression_over_time.png"
    )
