# SKILL.md

## Purpose

Use this skill when planning, implementing, modifying, testing, or verifying the mixed-autonomy traffic ABM.

The model tests whether aggressive autonomous vehicles produce a shared human norm shift, rejection, or polarization depending on how human drivers respond to AV behavior.

This skill preserves project-specific assumptions across prompts so the implementation does not drift.

## Core model idea

This is a toy ABM, not a realistic traffic simulator.

The model asks whether repeated exposure to aggressive AVs changes human baseline driving aggression. Humans differ in how they interpret AV behavior:

* **Assimilators** treat AV behavior as legitimate norm information.
* **Discounters** ignore AV behavior.
* **Rejecters** view aggressive AV behavior as negative outgroup behavior and contrast away from it.

The key scientific question is whether aggressive AV exposure changes the population mean, increases polarization, or has little effect.

## Model assumptions

* Human drivers have `baseline_aggression` in `[0, 1]`.
* Higher aggression means a more assertive or aggressive driving style.
* Human drivers have `susceptibility`, controlling how much they update after observations.
* Human drivers have an AV response type: `assimilator`, `discounter`, or `rejecter`.
* AVs have fixed high aggression, usually `0.90`.
* AVs never update.
* The model uses dynamic random mixing.
* At each timestep, agents are randomly paired into temporary encounters.
* Pairings dissolve after each timestep and are redrawn at the next timestep.
* Do not add lanes, crashes, routing, speed physics, or road geometry unless explicitly requested.

## Default parameters

Use these defaults unless the prompt explicitly changes them:

```text
N_agents = 200
timesteps = 100
seeds = 30

AV_aggression = 0.90
human_influence_weight = 1.00

initial_human_aggression ~ Normal(0.35, 0.08), clipped to [0, 1]
human_susceptibility ~ Uniform(0.02, 0.12)
```

## AV response types

Use these AV influence weights:

```text
assimilator: +0.50
discounter:   0.00
rejecter:    -0.50
```

Use these population environments:

| Environment         | Assimilators | Discounters | Rejecters |
| ------------------- | -----------: | ----------: | --------: |
| Mostly assimilation |          70% |         20% |       10% |
| Mixed               |          45% |         10% |       45% |
| Mostly rejection    |          10% |         20% |       70% |

Use these AV prevalence levels:

```text
0.00
0.25
0.50
```

## Update rules

### Human observes another human

When two humans are paired, both update simultaneously using pre-update baseline aggression values.

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

* positive `av_influence_weight`: move toward the AV
* zero `av_influence_weight`: ignore the AV
* negative `av_influence_weight`: move away from the AV

Always clip human aggression to `[0, 1]` after updating.

```text
new_baseline = min(1, max(0, new_baseline))
```

### AV update rule

AVs never update.

Do not modify AV aggression during the simulation.

If two AVs are paired, no update occurs.

## Timestep and trajectory rules

Always use this timestep convention:

* Record the initial state as `timestep = 0`.
* The first updated state must be `timestep = 1`.
* The final updated state must be `timestep = timesteps`.
* A run with `timesteps = 100` must produce 101 trajectory rows.
* Timestep values must be unique within each run.

Never allow the initial state and first update to both be labeled as timestep 0.

## Encounter metric rules

Keep human behavior metrics separate from AV behavior.

Preferred metrics:

* `mean_human_aggression`
* `var_human_aggression`
* `mean_human_hh_encounter_aggression`
* `mean_human_ha_encounter_aggression`
* `mean_av_ha_encounter_aggression`, optional diagnostic

Do not interpret a metric that averages human aggression and fixed AV aggression together as a human behavioral outcome.

The key diagnostic for norm shift is whether human-human encounter aggression changes after AV exposure.

## Main outcomes

Track and save:

* mean human baseline aggression over time
* variance in human baseline aggression over time
* final mean human baseline aggression
* final variance in human baseline aggression
* mean baseline aggression by AV response type
* mean human aggression during human-human encounters
* mean human aggression during human-AV encounters
* summary statistics across seeds

## Interpretation rules

A **shared norm shift** is supported if mean human baseline aggression increases over time, especially in the mostly-assimilation environment.

**Rejection** is supported if aggressive AV exposure produces little increase or a decrease in mean baseline aggression, especially in the mostly-rejection environment.

**Polarization** is supported if variance in human baseline aggression increases, especially in the mixed environment.

Do not interpret one simulation run as the result. Always summarize across seeds.

The strongest evidence for a broader norm shift is not only higher aggression during AV encounters. It is higher aggression in human-human encounters after AV exposure.

## Runtime verification rules

Use verification functions to check model logic before interpreting results.

Required checks:

* all human aggression values stay in `[0, 1]`
* population size stays constant
* AV aggression never changes
* same seed reproduces the same trajectory
* assimilators move toward AV aggression
* discounters do not update from AVs
* rejecters move away from AV aggression
* 0% AV condition behaves as a null condition
* each condition has the expected number of seeds
* each trajectory has unique timesteps
* each trajectory has exactly `timesteps + 1` rows
* each trajectory has timesteps from `0` to `timesteps`

If any core check fails, do not interpret the results.

## Pytest test rules

In addition to runtime verification, include formal pytest tests.

Use this structure:

```text
tests/
  test_agents.py
  test_model.py
  test_verification.py
  test_experiments.py
```

### Required test coverage

`test_agents.py` should test:

* clipping below 0 and above 1
* response-type assignment counts
* AV influence weights
* fixed AV initialization

`test_model.py` should test:

* simultaneous human-human updating from pre-update values
* human-AV updates affect only the human
* AVs never update
* timestep indexing is `0, 1, ..., timesteps`
* no duplicate timesteps
* trajectory length is `timesteps + 1`
* same seed gives identical trajectories

`test_verification.py` should test:

* bounds check passes for valid values
* bounds check fails for invalid values
* response-type micro-test passes
* timestep verification catches duplicate timesteps

`test_experiments.py` should test:

* experiment grid creates expected condition counts
* hierarchical seeds are deterministic
* aggregation reports expected seed counts

All tests should run with:

```bash
pytest
```

## Dependency rules

Every imported external package must appear in `requirements.txt`.

Minimum expected packages:

```text
numpy
pandas
matplotlib
pytest
```

The Dockerfile must install packages through `requirements.txt`:

```bash
pip install --no-cache-dir -r requirements.txt
```

Do not rely on local virtual environments, cached packages, or packages manually installed outside `requirements.txt`.

## Coding conventions

Use Python.

Prefer:

* plain Python
* NumPy
* pandas
* matplotlib
* pytest for formal tests

Keep the model simple and inspectable.

Use modular files:

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

`python run_simulation.py` must run the full simulation and save outputs to `results/`.

`pytest` must run formal tests.

Avoid hidden global state. Pass parameters explicitly where possible.

Use deterministic random seeds.

Save raw trajectories and summary results as CSV files.

## Visualization standards

Use `matplotlib` for all saved figures.

Figures should be clean, readable, and consistent across the project.

### General style

* Use concise, legible titles.
* Use clear axis labels with units or scale when relevant.
* Use consistent font sizes across plots.
* Avoid crowded legends.
* Avoid decorative effects.
* Use a light background.
* Save figures at high resolution.

Recommended defaults:

```python
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
```

### Color rules

Use colorblind-friendly colors.

Recommended Okabe-Ito style palette:

```python
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
```

Use the same colors consistently across figures.

Suggested mappings:

```text
Mostly assimilation: blue
Mixed: orange
Mostly rejection: green

0% AVs: gray or black
25% AVs: blue
50% AVs: red/orange
```

Do not rely only on color. Use line styles or markers when comparing multiple conditions.

### Required plots

At minimum, create:

1. Mean human baseline aggression over time
2. Variance in human baseline aggression over time
3. Final mean aggression by condition
4. Final variance or polarization by condition
5. Human-human encounter aggression over time

Optional useful plots:

* mean aggression by response type
* human-AV human aggression over time
* distribution of final baseline aggression

### Plot titles

Use short titles that state the main comparison.

Good examples:

```text
Mean aggression over time
Variance in aggression over time
Final aggression by AV prevalence
Polarization under mixed AV responses
Human-human aggression after AV exposure
```

Avoid vague or overly long titles.

## Common failure modes

Watch for:

* accidentally updating AV aggression
* treating all humans as the same AV response type
* forgetting to clip aggression to `[0, 1]`
* recording timestep 0 twice
* producing duplicate timesteps
* reporting one seed as if it were the result
* averaging human and AV aggression into one ambiguous human-AV outcome
* interpreting a mean shift as polarization
* failing to save results reproducibly
* forgetting to include imported packages in `requirements.txt`
* letting the implementation drift into a traffic simulator instead of a toy ABM
* creating plots with inconsistent colors or unreadable labels

## Implementation discipline

When editing code, make one bounded change at a time.

After each major change, run the relevant tests and verification checks.

Do not interpret simulation results until all runtime checks and pytest tests pass.
