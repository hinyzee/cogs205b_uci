# PROMPT.md

## Project goal

Build a small, inspectable, and reproducible agent-based model of how human drivers adapt in a mixed-autonomy traffic environment where aggressive autonomous vehicles are present.

The model should examine a motivating scenario where human drivers repeatedly encounter self-driving cars that behave differently from normal human traffic. Some autonomous vehicles may be overly cautious, hesitate too much, or interrupt the flow of traffic, which can create frustration. Other self-driving systems may behave in a way that feels more assertive or aggressive, such as taking smaller gaps, accelerating more quickly, or yielding less often.

The central question is not simply whether aggressive behavior spreads. That would be too obvious if the model assumes humans automatically copy what they see. Instead, the model should ask when aggressive AV behavior becomes a source of norm information and when it is rejected.

In this model, people may respond to aggressive AVs in different ways. Some drivers may treat aggressive AV behavior as evidence that the local traffic norm has become more aggressive, so they update toward it. Other drivers may discount AV behavior because they see self-driving cars as strange, non-human, or irrelevant to how humans should drive. Still others may actively reject the behavior because they do not want to be identified with aggressive self-driving EVs or because they view those vehicles negatively. In that case, exposure to aggressive AVs could make them contrast away from the AV style rather than imitate it.

This means aggressive AV exposure could have different population-level effects. If most humans treat AVs as legitimate norm-setters, aggressive AVs may produce a shared upward shift in human driving aggression. If most humans discount or reject AVs, aggressive AV prevalence may have little effect or may even reduce human aggression. If the population is mixed, aggressive AV exposure may increase polarization: some humans become more aggressive, while others become less aggressive.

The goal of the ABM is to compare these possible mechanisms in a simple theoretical environment. The model is not intended to be a realistic traffic simulator. It should not model lanes, crashes, road geometry, routing, or travel time. Instead, it should isolate repeated exposure, social updating, and source-dependent influence as the core mechanisms.

Before implementation, produce a detailed implementation plan that can be curated and saved as `PLAN.md`.

## Research question

Can aggressive autonomous vehicles shift human driving norms, or does their effect depend on whether humans treat AV behavior as legitimate, irrelevant, or negative outgroup behavior?

## Main hypothesis

Aggressive AV exposure will not have one uniform effect on human drivers.

If most humans treat AVs as legitimate norm-setters, higher AV prevalence should increase mean human baseline aggression.

If most humans discount or reject AV behavior, higher AV prevalence should produce little mean increase or may even decrease aggression.

If the population is mixed, higher AV prevalence may increase variance in human baseline aggression, because assimilators move toward aggressive AVs while rejecters move away from them.

## Model overview

The model should use a dynamic random-mixing topology.

At each timestep:

1. all agents are randomly shuffled
2. agents are paired into temporary encounters
3. humans observe the aggression of the paired agent
4. all updates are computed from pre-update aggression values
5. updates are applied after all pair updates are computed
6. pairings dissolve and are redrawn next timestep

This topology is a simplifying assumption. It treats AV exposure as a function of AV prevalence and repeated encounters rather than physical road structure.

## Agent types

There are two types of agents: human drivers and aggressive AVs.

### Human agents

Each human has:

* `baseline_aggression`: continuous value in `[0, 1]`

  * represents the human’s normal driving style
  * higher values mean more aggressive driving
* `susceptibility`: continuous value in `[0, 1]`

  * controls how strongly the human updates after an observation
* `av_response_type`

  * `assimilator`: treats AV behavior as legitimate norm information
  * `discounter`: ignores AV behavior
  * `rejecter`: treats aggressive AV behavior as negative outgroup behavior
* `av_influence_weight`

  * `+0.50` for assimilators
  * `0.00` for discounters
  * `-0.50` for rejecters

### AV agents

Each AV has:

* fixed aggression value, such as `0.90`
* no susceptibility
* no learning
* no response type
* aggression stays constant across the simulation

AVs are fixed high-aggression agents.

## Initial values

Use the following default values unless there is a clear reason to revise them:

* `N_agents = 200`
* `timesteps = 100`
* `seeds = 30`
* `AV_aggression = 0.90`
* `human_influence_weight = 1.00`

Initialize human baseline aggression as:

```text
initial_human_aggression ~ Normal(0.35, 0.08), clipped to [0, 1]
```

Initialize human susceptibility as:

```text
susceptibility ~ Uniform(0.02, 0.12)
```

## Update rules

### Human observes another human

When two humans are paired, both humans update simultaneously using their pre-update baselines.

```text
new_i = old_i + susceptibility_i * human_influence_weight * (old_j - old_i)

new_j = old_j + susceptibility_j * human_influence_weight * (old_i - old_j)
```

Use:

```text
human_influence_weight = 1.00
```

### Human observes an AV

When a human is paired with an AV, only the human updates.

```text
new_baseline = old_baseline
             + susceptibility * av_influence_weight * (AV_aggression - old_baseline)
```

Interpretation:

* positive `av_influence_weight`: human moves toward the aggressive AV
* zero `av_influence_weight`: human ignores the AV
* negative `av_influence_weight`: human moves away from the aggressive AV

After every update, clip human aggression to `[0, 1]`.

```text
new_baseline = min(1, max(0, new_baseline))
```

### AVs

AVs never update.

If two AVs are paired, no update occurs.

## Timestep convention

Use this convention exactly.

* The initial state is recorded as `timestep = 0`.
* The first updated state is recorded as `timestep = 1`.
* The final updated state is recorded as `timestep = timesteps`.
* If `timesteps = 100`, each trajectory must contain exactly 101 rows: `0, 1, 2, ..., 100`.
* There must be no duplicate timestep values within a run.

Do not record the initial state and first updated state using the same timestep.

## Experimental conditions

The simulation should cross two factors.

### Factor 1: AV prevalence

Use:

* `0.00`
* `0.25`
* `0.50`

These represent 0%, 25%, and 50% aggressive AVs in the population.

### Factor 2: human AV-response environment

Use three population environments:

| Environment         | Assimilators | Discounters | Rejecters |
| ------------------- | -----------: | ----------: | --------: |
| Mostly assimilation |          70% |         20% |       10% |
| Mixed               |          45% |         10% |       45% |
| Mostly rejection    |          10% |         20% |       70% |

The 0% AV condition serves as a null check. AV-response type should not matter much when there are no AVs.

## Main outcomes

Track and save the following outcomes for each simulation run:

* mean human baseline aggression over time
* variance in human baseline aggression over time
* final mean human baseline aggression
* final variance in human baseline aggression
* mean baseline aggression by AV response type
* mean human aggression in human-human encounters
* mean human aggression in human-AV encounters
* fixed AV aggression in human-AV encounters, if useful as a diagnostic
* summary statistics across seeds

Do not average human aggression and fixed AV aggression into one ambiguous human-AV encounter metric. Keep human behavior metrics separate from AV behavior.

## Key diagnostics

A shared norm shift is supported if mean human baseline aggression increases over time, especially in the mostly-assimilation condition.

Rejection is supported if aggressive AV exposure produces little increase or a decrease in mean human baseline aggression, especially in the mostly-rejection condition.

Polarization is supported if variance in human baseline aggression increases, especially in the mixed condition. This would mean assimilators become more aggressive while rejecters move away from the aggressive AV norm.

The strongest evidence for a broader norm shift is not just higher aggression during AV encounters. It is higher aggression in later human-human encounters after AV exposure.

## Verification checks

The implementation must include simulation-specific verification checks.

Include runtime verification functions for:

1. **Bounds check**

   * all human aggression values remain in `[0, 1]`

2. **Population check**

   * the number of humans and AVs remains constant across timesteps

3. **Seed determinism**

   * running the same condition with the same seed produces the same trajectory

4. **Null check**

   * with 0% AVs, AV prevalence cannot drive changes in aggression

5. **AV update check**

   * AV aggression remains fixed across the simulation

6. **Response-type check**

   * assimilators move toward AV aggression after AV encounters
   * discounters do not update from AVs
   * rejecters move away from AV aggression after AV encounters

7. **Sensitivity check**

   * results are summarized across multiple seeds, not based on a single run

8. **Timestep check**

   * each trajectory has unique timesteps
   * each trajectory has exactly `timesteps + 1` rows
   * timesteps run from `0` to `timesteps`

The project should fail loudly if core verification checks fail.

## Test files

In addition to runtime verification, create formal pytest tests.

Use this structure:

```text
tests/
  test_agents.py
  test_model.py
  test_verification.py
  test_experiments.py
```

### Required tests

`tests/test_agents.py` should test:

* `clip_aggression()` clips values below 0 and above 1
* response type assignment produces the expected counts
* AV influence weights match the intended values
* AVs are initialized with fixed aggression

`tests/test_model.py` should test:

* human-human updates are simultaneous and use pre-update baselines
* human-AV updates affect only the human
* AVs never update
* timestep indexing is `0, 1, ..., timesteps`
* no duplicate timesteps are produced
* a run with `timesteps = T` produces `T + 1` trajectory rows
* same seed produces identical trajectories

`tests/test_verification.py` should test:

* bounds check passes for valid aggression values
* bounds check fails for out-of-range human aggression
* response-type micro-test passes
* timestep check catches duplicate timesteps

`tests/test_experiments.py` should test:

* experiment grid produces the expected number of conditions
* deterministic hierarchical seeds are stable
* aggregation includes the expected number of seeds per condition

These tests should be runnable with:

```bash
pytest
```

## Requirements and Docker

Create a `requirements.txt` file containing every external package used by the project.

At minimum include:

```text
numpy
pandas
matplotlib
pytest
```

The Dockerfile must install from `requirements.txt` using:

```bash
pip install --no-cache-dir -r requirements.txt
```

Do not rely on local virtual environments, cached packages, or manually installed packages outside `requirements.txt`.

The Docker image should run:

```bash
python run_simulation.py
```

It is acceptable if tests are run separately with:

```bash
pytest
```

## Required project structure

Create an `abm-project/` folder with multiple files:

```text
abm-project/
  PROMPT.md
  PLAN.md
  SKILL.md
  README.md
  Dockerfile
  requirements.txt
  run_simulation.py
  src/
    __init__.py
    agents.py
    model.py
    experiments.py
    verification.py
    plotting.py
  tests/
    test_agents.py
    test_model.py
    test_verification.py
    test_experiments.py
  results/
```

## Implementation requirements

The implementation should include:

* `PROMPT.md` containing this planning prompt
* `PLAN.md` containing the curated implementation plan
* `SKILL.md` as the context-management artifact
* ABM implementation code
* pytest test files
* `run_simulation.py`, which reproducibly runs the simulation and saves results
* a `results/` folder with CSV outputs and plots
* a `Dockerfile` that reproduces the environment
* `README.md` with:

  * model specification
  * results
  * reflection on accuracy and trust

The project should run with:

```bash
python run_simulation.py
```

The tests should run with:

```bash
pytest
```

## Context-management artifact

Create a project-specific `SKILL.md` that helps preserve model assumptions and coding constraints across prompts.

The `SKILL.md` should include:

* model assumptions
* update rules
* timestep convention
* allowed parameter values
* verification rules
* testing rules
* visualization standards
* coding conventions
* common failure modes

It should not just restate generic instructions. It should be useful for maintaining consistency during implementation.

## Coding requirements

Use Python.

Keep the implementation simple, readable, and modular.

Use deterministic random seeds.

Save raw trajectories and summary results to CSV files.

Save plots to the `results/` folder.

Use `matplotlib` for all plots.

Avoid unnecessary complexity. The model should be small enough to explain and verify.

## Planning instruction

Before implementation, produce a detailed implementation plan that can be saved as `PLAN.md`.

The plan should include:

1. scientific model summary
2. file-by-file implementation plan
3. data structures
4. functions/classes to implement
5. parameter grid
6. update logic
7. timestep convention
8. verification functions
9. pytest test plan
10. expected outputs
11. plotting plan
12. README outline
13. requirements and Dockerfile plan
14. possible risks or ambiguities to resolve before coding

The plan should be specific enough that it can be saved as `PLAN.md` and used to implement the model.
