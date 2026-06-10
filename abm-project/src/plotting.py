"""Matplotlib plotting utilities for simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

PREVALENCE_COLORS = {
    0.0: COLORS["black"],
    0.25: COLORS["blue"],
    0.5: COLORS["red"],
}

PREVALENCE_STYLES = {
    0.0: {"linestyle": "-", "marker": "o"},
    0.25: {"linestyle": "--", "marker": "s"},
    0.5: {"linestyle": "-.", "marker": "^"},
}

ENV_LABELS = {
    "mostly_assimilation": "Mostly assimilation",
    "mixed": "Mixed",
    "mostly_rejection": "Mostly rejection",
}

PREVALENCE_LABELS = {
    0.0: "0% AV",
    0.25: "25% AV",
    0.5: "50% AV",
}

ENVIRONMENT_ORDER = ["mostly_assimilation", "mixed", "mostly_rejection"]
PREVALENCE_ORDER = [0.0, 0.25, 0.5]


def _apply_style() -> None:
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
        }
    )


def _prevalence_pct(av_prevalence: float) -> str:
    return f"{int(av_prevalence * 100)}%"


def _plot_grouped_bar(
    final_summary_df: pd.DataFrame,
    *,
    value_col: str,
    error_col: str | None,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Grouped bar chart: x = AV prevalence, color = environment."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    n_envs = len(ENVIRONMENT_ORDER)
    x = np.arange(len(PREVALENCE_ORDER))
    bar_width = 0.8 / n_envs

    for env_idx, environment in enumerate(ENVIRONMENT_ORDER):
        env_df = final_summary_df[
            final_summary_df["environment"] == environment
        ].set_index("av_prevalence")

        values = [env_df.loc[p, value_col] for p in PREVALENCE_ORDER]
        x_positions = x + (env_idx - (n_envs - 1) / 2) * bar_width

        error_kw = {"capsize": 3, "elinewidth": 1}
        if error_col is not None and error_col in final_summary_df.columns:
            errors = [env_df.loc[p, error_col] for p in PREVALENCE_ORDER]
            ax.bar(
                x_positions,
                values,
                width=bar_width,
                label=ENV_LABELS[environment],
                color=ENV_COLORS[environment],
                alpha=0.85,
                yerr=errors,
                error_kw=error_kw,
            )
        else:
            ax.bar(
                x_positions,
                values,
                width=bar_width,
                label=ENV_LABELS[environment],
                color=ENV_COLORS[environment],
                alpha=0.85,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_prevalence_pct(p) for p in PREVALENCE_ORDER])
    ax.set_xlabel("AV prevalence")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Environment", loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_faceted_timeseries(
    summary_df: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Three-panel time series: columns = environment, lines = AV prevalence."""
    _apply_style()
    fig, axes = plt.subplots(
        1,
        len(ENVIRONMENT_ORDER),
        figsize=(13, 4.5),
        sharey=True,
    )

    legend_handles: list = []
    legend_labels: list[str] = []

    for ax, environment in zip(axes, ENVIRONMENT_ORDER):
        env_df = summary_df[summary_df["environment"] == environment]

        for av_prevalence in PREVALENCE_ORDER:
            group = env_df[env_df["av_prevalence"] == av_prevalence].sort_values(
                "timestep"
            )
            style = PREVALENCE_STYLES[av_prevalence]
            (line,) = ax.plot(
                group["timestep"],
                group[value_col],
                label=PREVALENCE_LABELS[av_prevalence],
                color=PREVALENCE_COLORS[av_prevalence],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=10,
                linewidth=1.5,
                markersize=4,
            )
            if environment == ENVIRONMENT_ORDER[0]:
                legend_handles.append(line)
                legend_labels.append(PREVALENCE_LABELS[av_prevalence])

        ax.set_title(ENV_LABELS[environment])
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.legend(
        legend_handles,
        legend_labels,
        title="AV prevalence",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(PREVALENCE_ORDER),
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_aggression_over_time(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean human baseline aggression over time by environment panel."""
    _plot_faceted_timeseries(
        summary_df,
        value_col="mean_human_aggression_mean",
        ylabel="Mean human aggression",
        title="Mean aggression over time",
        output_path=output_path,
    )


def plot_variance_over_time(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot variance in human baseline aggression over time by environment panel."""
    _plot_faceted_timeseries(
        summary_df,
        value_col="var_human_aggression_mean",
        ylabel="Variance in human aggression",
        title="Polarization over time",
        output_path=output_path,
    )


def plot_final_mean_by_condition(
    final_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final mean aggression grouped by AV prevalence and environment."""
    _plot_grouped_bar(
        final_summary_df,
        value_col="final_mean_human_aggression",
        error_col="final_std_human_aggression",
        ylabel="Final mean aggression",
        title="Final aggression by AV prevalence",
        output_path=output_path,
    )


def plot_final_variance_by_condition(
    final_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final variance grouped by AV prevalence and environment."""
    _plot_grouped_bar(
        final_summary_df,
        value_col="final_mean_variance",
        error_col="final_std_variance",
        ylabel="Final variance in aggression",
        title="Final polarization by AV prevalence",
        output_path=output_path,
    )


def plot_human_human_encounter_aggression(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot human-human encounter aggression over time by environment panel."""
    _plot_faceted_timeseries(
        summary_df,
        value_col="mean_human_hh_encounter_aggression_mean",
        ylabel="Mean human-human aggression",
        title="Human-human aggression after AV exposure",
        output_path=output_path,
    )


def generate_all_plots(
    summary_by_timestep: pd.DataFrame,
    summary_final: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generate all required plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_aggression_over_time(
        summary_by_timestep,
        plots_dir / "mean_aggression_over_time.png",
    )
    plot_variance_over_time(
        summary_by_timestep,
        plots_dir / "variance_over_time.png",
    )
    plot_final_mean_by_condition(
        summary_final,
        plots_dir / "final_mean_by_condition.png",
    )
    plot_final_variance_by_condition(
        summary_final,
        plots_dir / "final_variance_by_condition.png",
    )
    plot_human_human_encounter_aggression(
        summary_by_timestep,
        plots_dir / "human_human_encounter_aggression.png",
    )
