# Mixed-Autonomy Traffic ABM

## Overview

This agent-based model examines whether aggressive autonomous vehicles (AVs) can shift human driving norms in a mixed-autonomy traffic environment. The central question is whether AV behavior is treated as legitimate norm information, irrelevant, or negative outgroup behavior — and how that shapes population-level aggression over repeated random encounters.

**Main hypothesis:** Aggressive AV exposure will not have one uniform effect. Assimilation-dominated populations should show rising mean aggression; rejection-dominated populations should show little increase or a decrease; mixed populations may show increased variance (polarization).

## Model specification

### Agent types

**Human drivers** have:
- `baseline_aggression` in [0, 1] — normal driving style
- `susceptibility` in [0.02, 0.12] — update strength after observations
- `av_response_type`: assimilator (+0.50 AV weight), discounter (0.00), or rejecter (-0.50)

**AV agents** have fixed aggression (default 0.90) and never update.

### Initial values

- `N_agents = 200`, `timesteps = 100`, `seeds = 30`
- Human aggression: `Normal(0.35, 0.08)` clipped to [0, 1]
- Susceptibility: `Uniform(0.02, 0.12)`

### Update rules

**Human observes human** (both update simultaneously):
```
new_baseline = old_baseline + susceptibility * 1.00 * (observed - old_baseline)
```

**Human observes AV** (human only):
```
new_baseline = old_baseline + susceptibility * av_influence_weight * (AV_aggression - old_baseline)
```

All values clipped to [0, 1] after updating.

### Pairing topology

Each timestep: shuffle all agents, pair sequentially (100 pairs). Pairings dissolve after the timestep.

### Experimental conditions

| Environment | Assimilators | Discounters | Rejecters |
|-------------|-------------:|------------:|----------:|
| Mostly assimilation | 70% | 20% | 10% |
| Mixed | 45% | 10% | 45% |
| Mostly rejection | 10% | 20% | 70% |

Crossed with AV prevalence: 0%, 25%, 50% (270 runs total at 30 seeds).

## How to run

**Local:**
```bash
pip install -r requirements.txt
python run_simulation.py
```

**With options:**
```bash
python run_simulation.py --seeds 30 --timesteps 100 --output-dir results
```

**Docker:**
```bash
docker build -t abm .
docker run -v $(pwd)/results:/app/results abm
```

## Results

After running, outputs are saved to `results/`:

- `results/trajectories/` — per-run timestep CSVs
- `results/summaries/run_summary.csv` — one row per simulation
- `results/summaries/aggregated_summary.csv` — seed-averaged final outcomes
- `results/summaries/aggregated_trajectories.csv` — seed-averaged time series
- `results/plots/` — PNG figures

Key plots:
- `mean_aggression_over_time.png`
- `variance_over_time.png`
- `final_mean_by_condition.png`
- `final_variance_by_condition.png`
- `hh_encounter_aggression_over_time.png`

Observed patterns (30 seeds, aggregated):
- **Shared norm shift:** mostly-assimilation final mean rises from 0.35 (0% AV) to 0.60 (50% AV)
- **Rejection:** mostly-rejection final mean falls from 0.35 (0% AV) to 0.06 (50% AV)
- **Polarization:** mixed environment final variance rises from ~0.000006 (0% AV) to ~0.046 (50% AV), while mean aggression decreases to 0.26

## Verification

The simulation runs these checks automatically:

| Check | Criterion |
|-------|-----------|
| Bounds | All human aggression in [0, 1] |
| Population | Human/AV counts constant |
| Seed determinism | Same seed → identical results |
| AV fixed | AV aggression never changes |
| Response types | Assimilator up, discounter flat, rejecter down after AV encounter |
| Null (0% AV) | Final means across environments differ by < 0.005 |
| Multi-seed | 30 seeds per condition |

`run_simulation.py` exits with code 1 if pre-grid checks (1–3, 5, 6) fail.

## Reflection on accuracy and trust

**Trust results when:**
- All verification checks pass
- Patterns are consistent across 30 seeds, not a single run
- AV prevalence and response environment show distinguishable effects
- Human-human encounter aggression rises after AV exposure in assimilation conditions (norm diffusion beyond direct AV encounters)

**Distrust results when:**
- Model logic fails (bounds violations, population changes, non-deterministic seeds)
- 0%, 25%, and 50% AV prevalence produce identical trajectories
- AV response type has no effect at high AV prevalence
- Conclusions rest on a single seed

**Limitations:** This is a toy ABM isolating exposure and social updating — not a traffic simulator. It uses random mixing rather than spatial structure, and simultaneous within-timestep updates.

## Project layout

```
abm-project/
  PROMPT.md           Planning prompt
  PLAN.md             Implementation plan
  SKILL.md            Context-management artifact
  README.md           This file
  Dockerfile          Reproducible environment
  requirements.txt    Python dependencies
  run_simulation.py   Entry point
  src/
    agents.py         Agent types and population factories
    model.py          Core simulation engine
    experiments.py    Factorial grid and CSV writers
    verification.py   Model verification checks
    plotting.py       Figure generation
  results/            Generated outputs (gitignored)
```
