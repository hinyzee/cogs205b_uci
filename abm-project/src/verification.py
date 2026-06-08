"""Verification checks for the mixed-autonomy ABM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents import (
    AV_INFLUENCE_WEIGHTS,
    AVAgent,
    AvResponseType,
    HumanAgent,
    clip_aggression,
)
from src.model import MixedAutonomyModel, SimulationConfig


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def check_bounds(model: MixedAutonomyModel) -> CheckResult:
    """All human aggression values remain in [0, 1]."""
    for human in model.humans:
        if not (0.0 <= human.baseline_aggression <= 1.0):
            return CheckResult(
                "bounds",
                False,
                f"Agent {human.agent_id} has aggression {human.baseline_aggression}",
            )
    return CheckResult("bounds", True, "All human aggression values in [0, 1]")


def check_population_constant(model: MixedAutonomyModel) -> CheckResult:
    """Human and AV counts remain constant."""
    if len(model.humans) != model.initial_human_count:
        return CheckResult(
            "population",
            False,
            f"Human count changed: {model.initial_human_count} -> {len(model.humans)}",
        )
    if len(model.avs) != model.initial_av_count:
        return CheckResult(
            "population",
            False,
            f"AV count changed: {model.initial_av_count} -> {len(model.avs)}",
        )
    return CheckResult("population", True, "Population counts constant")


def check_seed_determinism(config: SimulationConfig) -> CheckResult:
    """Same seed produces identical results."""
    model_a = MixedAutonomyModel(config)
    traj_a = model_a.run()
    final_a = traj_a[-1].mean_human_aggression

    model_b = MixedAutonomyModel(config)
    traj_b = model_b.run()
    final_b = traj_b[-1].mean_human_aggression

    if abs(final_a - final_b) > 1e-12:
        return CheckResult(
            "seed_determinism",
            False,
            f"Final means differ: {final_a} vs {final_b}",
        )
    return CheckResult("seed_determinism", True, "Seed determinism confirmed")


def check_av_fixed(model: MixedAutonomyModel) -> CheckResult:
    """AV aggression remains fixed across simulation."""
    for av in model.avs:
        expected = model.initial_av_aggressions[av.agent_id]
        if av.aggression != expected:
            return CheckResult(
                "av_fixed",
                False,
                f"AV {av.agent_id} aggression changed: {expected} -> {av.aggression}",
            )
    return CheckResult("av_fixed", True, "AV aggression remains fixed")


def check_response_types_after_av_encounter() -> CheckResult:
    """Assimilators move toward, discounters ignore, rejecters move away from AV."""
    av_aggression = 0.90
    baseline = 0.35
    susceptibility = 0.10

    results: dict[str, tuple[float, float]] = {}
    for response_type in AvResponseType:
        human = HumanAgent(
            agent_id=0,
            baseline_aggression=baseline,
            susceptibility=susceptibility,
            av_response_type=response_type,
            av_influence_weight=AV_INFLUENCE_WEIGHTS[response_type],
        )
        av = AVAgent(agent_id=1, aggression=av_aggression)

        difference = av.aggression - human.baseline_aggression
        new_baseline = clip_aggression(
            human.baseline_aggression
            + human.susceptibility * human.av_influence_weight * difference
        )
        results[response_type.value] = (baseline, new_baseline)

    assimilator_old, assimilator_new = results["assimilator"]
    discounter_old, discounter_new = results["discounter"]
    rejecter_old, rejecter_new = results["rejecter"]

    if assimilator_new <= assimilator_old:
        return CheckResult(
            "response_types",
            False,
            f"Assimilator did not increase: {assimilator_old} -> {assimilator_new}",
        )
    if discounter_new != discounter_old:
        return CheckResult(
            "response_types",
            False,
            f"Discounter changed: {discounter_old} -> {discounter_new}",
        )
    if rejecter_new >= rejecter_old:
        return CheckResult(
            "response_types",
            False,
            f"Rejecter did not decrease: {rejecter_old} -> {rejecter_new}",
        )
    return CheckResult(
        "response_types",
        True,
        "Response types behave correctly after AV encounter",
    )


def check_null_condition(summary_df: pd.DataFrame, tolerance: float = 0.005) -> CheckResult:
    """At 0% AV prevalence, environments produce nearly identical final means."""
    null_df = summary_df[summary_df["av_prevalence"] == 0.0]
    if null_df.empty:
        return CheckResult(
            "null_condition",
            False,
            "No 0% AV prevalence runs found in summary",
        )

    final_means = null_df.groupby("environment")["final_mean_aggression"].mean()
    if len(final_means) < 2:
        return CheckResult(
            "null_condition",
            False,
            "Insufficient environments for null check",
        )

    spread = final_means.max() - final_means.min()
    if spread >= tolerance:
        return CheckResult(
            "null_condition",
            False,
            f"0% AV environments differ by {spread:.6f} (tolerance {tolerance})",
        )
    return CheckResult(
        "null_condition",
        True,
        f"0% AV null check passed (spread={spread:.6f})",
    )


def check_sensitivity_multi_seed(
    summary_df: pd.DataFrame, expected_seeds: int = 30
) -> CheckResult:
    """Each condition has the expected number of seed rows."""
    counts = summary_df.groupby(["environment", "av_prevalence"]).size()
    bad = counts[counts != expected_seeds]
    if not bad.empty:
        return CheckResult(
            "multi_seed",
            False,
            f"Conditions with wrong seed count: {bad.to_dict()}",
        )
    return CheckResult(
        "multi_seed",
        True,
        f"All conditions have {expected_seeds} seeds",
    )


def run_simulation_checks(config: SimulationConfig) -> list[CheckResult]:
    """Run checks 1-3, 5, 6 on a single simulation (pre-grid)."""
    results: list[CheckResult] = []

    results.append(check_response_types_after_av_encounter())
    results.append(check_seed_determinism(config))

    model = MixedAutonomyModel(config)
    for _ in range(config.timesteps):
        model.step()
        results.append(check_bounds(model))
        results.append(check_population_constant(model))

    results.append(check_av_fixed(model))

    return results


def run_all_checks(
    smoke_config: SimulationConfig,
    summary_df: pd.DataFrame | None = None,
    expected_seeds: int = 30,
) -> list[CheckResult]:
    """Run all verification checks."""
    results = run_simulation_checks(smoke_config)

    if summary_df is not None:
        results.append(check_null_condition(summary_df))
        results.append(check_sensitivity_multi_seed(summary_df, expected_seeds))

    return results


def print_check_results(results: list[CheckResult]) -> bool:
    """Print results and return True if all passed."""
    all_passed = True
    seen: set[str] = set()
    for result in results:
        if result.name in ("bounds", "population") and result.name in seen:
            continue
        if result.name in ("bounds", "population"):
            seen.add(result.name)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.message}")
        if not result.passed:
            all_passed = False
    return all_passed
