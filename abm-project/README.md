# Mixed-Autonomy Traffic ABM

Toy agent-based model asking whether repeated exposure to aggressive autonomous vehicles (AVs) shifts human baseline driving aggression, and whether that effect depends on how humans interpret AV behavior.

At each timestep, 200 agents are randomly shuffled and paired into temporary encounters. Humans update baseline aggression after observing a partner. This is not a traffic simulator: there are no lanes, crashes, routing, or travel-time physics.

## Model specification

### Agent types and parameters

| Parameter | Default value |
|-----------|---------------|
| `N_agents` | 200 |
| `timesteps` | 100 |
| `seeds` | 30 |
| `AV_aggression` | 0.90 |
| `human_influence_weight` | 1.00 |
| Initial human aggression | `Normal(0.35, 0.08)`, clipped to `[0, 1]` |
| Human susceptibility | `Uniform(0.02, 0.12)` |

**Human agents** have `baseline_aggression`, `susceptibility`, and an AV response type:

| Response type | `av_influence_weight` | Behavior |
|---------------|----------------------:|----------|
| Assimilator | +0.50 | Moves toward aggressive AV behavior |
| Discounter | 0.00 | Ignores AV behavior |
| Rejecter | −0.50 | Contrasts away from aggressive AV behavior |

**AV agents** have fixed aggression (0.90), no susceptibility, and never update.

### Update rules

**Human observes human** (simultaneous, pre-update values):

```text
new_i = old_i + susceptibility_i * human_influence_weight * (old_j - old_i)
new_j = old_j + susceptibility_j * human_influence_weight * (old_i - old_j)
```

**Human observes AV** (human only updates):

```text
new_baseline = old_baseline + susceptibility * av_influence_weight * (AV_aggression - old_baseline)
```

All human aggression values are clipped to `[0, 1]` after each update.

### Timestep convention

- `timestep = 0`: initial state (no updates applied)
- `timestep = 1 … timesteps`: state after each update round
- Each run produces exactly `timesteps + 1` rows (101 rows when `timesteps = 100`)

### Experimental grid (3 × 3)

**AV prevalence:** 0%, 25%, 50%

**Human AV-response environments:**

| Environment | Assimilators | Discounters | Rejecters |
|-------------|-------------:|------------:|----------:|
| Mostly assimilation | 70% | 20% | 10% |
| Mixed | 45% | 10% | 45% |
| Mostly rejection | 10% | 20% | 70% |

Total: 9 conditions × 30 seeds = 270 simulation runs.

### Encounter metrics (separated)

Human and AV behavior are tracked separately in `encounter_metrics.csv`:

| Metric | Description |
|--------|-------------|
| `mean_human_hh_encounter_aggression` | Mean human aggression in human–human pairs (pre-update) |
| `mean_human_ha_encounter_aggression` | Mean human aggression in human–AV pairs (pre-update) |
| `mean_av_ha_encounter_aggression` | Fixed AV aggression in human–AV pairs (diagnostic only) |

## How to run

```bash
pip install -r requirements.txt
python3 run_simulation.py
pytest
```

Docker: `docker build -t mixed-autonomy-abm .` then `docker run --rm mixed-autonomy-abm`.

Outputs are written to `results/` (CSVs and `results/plots/*.png`).

## Results

Values below come from `results/summary_final.csv`, aggregated across 30 seeds per condition at the final timestep (`timestep = 100`). See `results/plots/` for figures.

| AV prevalence | Environment | Final mean aggression | Final variance | Final human–human encounter mean |
|--------------:|-------------|----------------------:|---------------:|---------------------------------:|
| 0% | Mostly assimilation | 0.351 | 0.000 | 0.351 |
| 0% | Mixed | 0.350 | 0.000 | 0.350 |
| 0% | Mostly rejection | 0.350 | 0.000 | 0.350 |
| 25% | Mostly assimilation | 0.531 | 0.003 | 0.530 |
| 25% | Mixed | 0.307 | 0.009 | 0.308 |
| 25% | Mostly rejection | 0.067 | 0.005 | 0.070 |
| 50% | Mostly assimilation | 0.613 | 0.018 | 0.613 |
| 50% | Mixed | 0.263 | 0.046 | 0.269 |
| 50% | Mostly rejection | 0.066 | 0.014 | 0.066 |

### Toy-model interpretation

**Null check (0% AV).** In this simulation, final mean aggression stays near 0.350 across all three environments, with variance near zero. Under this toy ABM, AV response type has little effect when no AVs are present.

**Mostly assimilation.** Final mean aggression rises with AV prevalence (0.351 → 0.531 → 0.613). In this simulation, the mostly-assimilation environment shows the upward pattern expected when humans treat AV behavior as norm information. Human–human encounter means track the same direction (0.351 → 0.530 → 0.613), suggesting the shift is visible beyond direct human–AV encounters.

**Mostly rejection.** Final mean aggression falls at 25% and 50% AV prevalence (0.350 → 0.067 → 0.066). Under this toy ABM, the mostly-rejection environment shows the flat-or-decreasing pattern expected when humans contrast away from aggressive AVs. Human–human encounter means follow a similar pattern (0.350 → 0.070 → 0.066).

**Mixed (polarization).** Final mean aggression is moderate at 50% AV (0.263), but final variance increases sharply (0.000 → 0.009 → 0.046). In this simulation, the mixed environment shows rising spread consistent with polarization: assimilators and rejecters update in opposite directions. See `plots/final_variance_by_condition.png` and `plots/variance_over_time.png`.

**Human–human encounters.** Where baseline aggression shifts, human–human encounter means move in the same direction (e.g., mostly assimilation at 50% AV: baseline 0.613, human–human encounters 0.613). Under this toy ABM, changes are not confined to human–AV encounters alone.

These patterns depend on the model's assumptions: random mixing rather than road structure, fixed aggressive AVs, source-dependent human updating rules, and no traffic physics or institutional constraints. They describe behavior inside this simulation, not real-world driving.

## Reflection

The model was implemented using bounded prompts in `PROMPT.md`, `PLAN.md`, and `SKILL.md` to keep assumptions stable across steps. Accuracy was checked with pytest tests (agents, model dynamics, verification helpers, experiment grid) and runtime verification after each full run: bounds checks, population constancy, seed determinism, null behavior at 0% AV prevalence, fixed AV aggression, response-type checks for assimilators/discounters/rejecters, timestep indexing, and multi-seed summaries (30 seeds per condition). Results are aggregated across seeds, not taken from single runs.

If tests and verification pass, the CSV outputs and plots can be trusted as faithful outputs of this toy model under its stated assumptions. They should not be interpreted as direct evidence about real human drivers or real traffic conditions.

## Project files

```
abm-project/
  PROMPT.md          Planning prompt
  PLAN.md            Implementation plan
  SKILL.md           Context-management artifact
  run_simulation.py  Main entry point
  src/               Model implementation
  tests/             Pytest suite
  results/           Generated outputs (gitignored)
```
