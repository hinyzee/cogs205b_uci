# SKILL.md

## Purpose

Use this skill when planning, implementing, modifying, or verifying the mixed-autonomy traffic ABM.

The model tests whether aggressive autonomous vehicles produce a shared human norm shift, rejection, or polarization depending on how human drivers respond to AV behavior.

The skill should preserve project-specific assumptions across prompts so the implementation does not drift.

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

Humans update toward other human drivers.

```text
difference = observed_aggression - old_baseline

new_baseline = old_baseline
             + susceptibility * human_influence_weight * difference
```

Use:

```text
human_influence_weight = 1.00
```

### Human observes an AV

Humans update according to their individual AV response type.

```text
difference = AV_aggression - old_baseline

new_baseline = old_baseline
             + susceptibility * av_influence_weight * difference
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

## Main outcomes

Track and save:

* mean human baseline aggression over time
* variance in human baseline aggression over time
* final mean human baseline aggression
* final variance in human baseline aggression
* mean baseline aggression by AV response type
* mean aggression during human-human encounters
* mean aggression during human-AV encounters
* summary statistics across seeds

## Interpretation rules

A **shared norm shift** is supported if mean human baseline aggression increases over time, especially in the mostly-assimilation environment.

**Rejection** is supported if aggressive AV exposure produces little increase or a decrease in mean baseline aggression, especially in the mostly-rejection environment.

**Polarization** is supported if variance in human baseline aggression increases, especially in the mixed environment.

Do not interpret one simulation run as the result. Always summarize across seeds.

The strongest evidence for a broader norm shift is not only higher aggression during AV encounters. It is higher aggression in human-human encounters after AV exposure.

## Required verification checks

Before trusting results, verify:

* all human aggression values remain in `[0, 1]`
* population size remains constant
* AV aggression remains fixed
* the same seed reproduces the same trajectory
* 0% AV prevalence behaves as a null condition
* assimilators move toward AV aggression after AV encounters
* discounters do not update from AV encounters
* rejecters move away from AV aggression after AV encounters
* results are summarized across multiple seeds

## You should distrust your results if...

You should distrust your results if any basic model logic fails.

Examples:

* AVs change aggression over time
* human aggression values leave `[0, 1]`
* population size changes across timesteps
* the same seed does not reproduce the same trajectory
* AV response type has no effect when AV prevalence is high
* 0%, 25%, and 50% AV prevalence produce nearly identical results in all environments
* results only appear for one random seed

## Coding conventions

Use Python.

Prefer:

* plain Python
* NumPy
* pandas
* matplotlib

Keep the model simple and inspectable.

Use modular files:

```text
abm-project/
  PROMPT.md
  PLAN.md
  SKILL.md
  README.md
  Dockerfile
  run_simulation.py
  src/
    agents.py
    model.py
    experiments.py
    verification.py
    plotting.py
  results/
```

`python run_simulation.py` must run the full simulation and save outputs to `results/`.

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

Example:

```text
solid line: mostly assimilation
dashed line: mixed
dotted line: mostly rejection
```

### Required plots

At minimum, create:

1. Mean human baseline aggression over time
2. Variance in human baseline aggression over time
3. Final mean aggression by condition
4. Final variance or polarization by condition

Optional useful plots:

* mean aggression by response type
* human-human encounter aggression over time
* distribution of final baseline aggression

### Plot titles

Use short titles that state the main comparison.

Good examples:

```text
Mean aggression over time
Variance in aggression over time
Final aggression by AV prevalence
Polarization under mixed AV responses
```

Avoid vague or overly long titles.

Bad examples:

```text
Simulation Results
Aggression Plot
Graph of Agent-Based Model Results Across Many Conditions
```

## Common failure modes

Watch for:

* accidentally updating AV aggression
* treating all humans as the same AV response type
* forgetting to clip aggression to `[0, 1]`
* confusing temporary encounter behavior with baseline aggression
* reporting one seed as if it were the result
* interpreting a mean shift as polarization
* failing to save results reproducibly
* letting the implementation drift into a traffic simulator instead of a toy ABM
* creating plots with inconsistent colors or unreadable labels

## Implementation discipline

When editing code, make one bounded change at a time.

After each major change, run the relevant verification checks.

Do not interpret simulation results until all checks pass.
