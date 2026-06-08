# PROMPT.md

## Project goal

Build a small, inspectable, and reproducible agent-based model of how human drivers adapt in a mixed-autonomy traffic environment where aggressive autonomous vehicles are present.

The model should examine a motivating scenario where human drivers repeatedly encounter self-driving cars that behave differently from normal human traffic. Some autonomous vehicles may be overly cautious, hesitate too much, or interrupt the flow of traffic, which can create frustration. Other self-driving systems may behave in a way that feels more assertive or aggressive, such as taking smaller gaps, accelerating more quickly, or yielding less often.

The central question is not simply whether aggressive behavior spreads. That would be too obvious if the model assumes humans automatically copy what they see. Instead, the model should ask when aggressive AV behavior becomes a source of norm information and when it is rejected.

In this model, people may respond to aggressive AVs in different ways. Some drivers may treat aggressive AV behavior as evidence that the local traffic norm has become more aggressive, so they update toward it. Other drivers may discount AV behavior because they see self-driving cars as strange, non-human, or irrelevant to how humans should drive. Still others may actively reject the behavior because they do not want to be identified with aggressive self-driving EVs or because they view those vehicles negatively. In that case, exposure to aggressive AVs could make them contrast away from the AV style rather than imitate it.

This means aggressive AV exposure could have different population-level effects. If most humans treat AVs as legitimate norm-setters, aggressive AVs may produce a shared upward shift in human driving aggression. If most humans discount or reject AVs, aggressive AV prevalence may have little effect or may even reduce human aggression. If the population is mixed, aggressive AV exposure may increase polarization: some humans become more aggressive, while others become less aggressive.

The goal of the ABM is to compare these possible mechanisms in a simple theoretical environment. The model is not intended to be a realistic traffic simulator. It should not model lanes, crashes, road geometry, routing, or travel time. Instead, it should isolate repeated exposure, social updating, and source-dependent influence as the core mechanisms.

Do not implement code yet. First produce a detailed implementation plan that can be curated and saved as `PLAN.md`.

## Research question

Can aggressive autonomous vehicles shift human driving norms, or does their effect depend on whether humans treat AV behavior as legitimate, irrelevant, or negative outgroup behavior?

## Main hypothesis

Aggressive AV exposure will not have one uniform effect on human drivers.

If most humans treat AVs as legitimate norm-setters, higher AV prevalence should increase mean human baseline aggression.

If most humans discount or reject AV behavior, higher AV prevalence should produce little mean increase or may even decrease aggression.

If the population is mixed, higher AV prevalence may increase variance in human baseline aggression, because assimilators move toward aggressive AVs while rejecters move away from them.

## Model overview

The model should use a dynamic random-mixing topology. At each timestep, agents are randomly paired into temporary encounters. Pairings dissolve after the timestep and are redrawn at the next timestep.

This topology is a simplifying assumption. It treats AV exposure as a function of AV prevalence and repeated encounters rather than physical road structure. The purpose is to test the behavioral updating mechanism in an inspectable way.

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

* positive `av_influence_weight`: human moves toward the aggressive AV
* zero `av_influence_weight`: human ignores the AV
* negative `av_influence_weight`: human moves away from the aggressive AV

After every update, clip aggression to `[0, 1]`.

```text
new_baseline = min(1, max(0, new_baseline))
```

### AVs

AVs never update.

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
* mean aggression during human-human encounters
* mean aggression during human-AV encounters
* summary statistics across seeds

## Key diagnostics

A shared norm shift is supported if mean human baseline aggression increases over time, especially in the mostly-assimilation condition.

Rejection is supported if aggressive AV exposure produces little increase or a decrease in mean human baseline aggression, especially in the mostly-rejection condition.

Polarization is supported if variance in human baseline aggression increases, especially in the mixed condition. This would mean assimilators become more aggressive while rejecters move away from the aggressive AV norm.

The strongest evidence for a broader norm shift is not just higher aggression during AV encounters. It is higher aggression in later human-human encounters after AV exposure.

## Verification checks

The implementation must include simulation-specific verification checks.

Include checks for:

1. **Bounds check**

   * all human aggression values remain in `[0, 1]`

2. **Population check**

   * the number of humans and AVs remains constant across timesteps

3. **Seed determinism**

   * running the same condition with the same seed produces identical results

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

## You should distrust your results if...

You should distrust your results if any of the basic model logic fails. For example, the results are not trustworthy if AVs change aggression over time, human aggression values leave the `[0, 1]` range, population size changes across timesteps, or the same seed does not reproduce the same trajectory.

You should also distrust the scientific result if the main manipulation does not matter. For example, if 0%, 25%, and 50% AV prevalence produce nearly identical trajectories, or if AV response type has no effect even when AV prevalence is high, then the model is not actually testing the intended mechanism.

Finally, be cautious if the result only appears for one random seed. The conclusion should depend on the pattern across multiple seeds, not on a single lucky simulation run.

## Required project structure

Create an `abm-project/` folder with multiple files:

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

## Implementation requirements

The implementation should include:

* `PROMPT.md` containing this planning prompt
* `PLAN.md` containing the curated implementation plan
* a project-specific `SKILL.md` used as a context-management artifact
* ABM implementation code
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

## Context-management artifact

Create a project-specific `SKILL.md` that helps preserve model assumptions and coding constraints across prompts.

The `SKILL.md` should include:

* model assumptions
* update rules
* allowed parameter values
* verification rules
* coding conventions
* common failure modes

It should not just restate generic instructions. It should be useful for maintaining consistency during implementation.

## Coding requirements

Use Python.

Keep the implementation simple, readable, and modular.

Use deterministic random seeds.

Save raw trajectories and summary results to CSV files.

Save plots to the `results/` folder.

Avoid unnecessary complexity. The model should be small enough to explain and verify.

## Planning instruction

Do NOT write code yet.

Produce a detailed implementation plan that includes:

1. scientific model summary
2. file-by-file implementation plan
3. data structures
4. functions/classes to implement
5. parameter grid
6. update logic
7. verification tests
8. expected outputs
9. plotting plan
10. README outline
11. Dockerfile/environment plan
12. possible risks or ambiguities to resolve before coding

The plan should be specific enough that it can be saved as `PLAN.md` and used to implement the model.
