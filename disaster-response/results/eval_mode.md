# Evaluation mode: deterministic argmax understates performance by ~40%

## The finding

Evaluating a trained policy with **argmax** actions — standard practice in much of RL — is
badly misleading in this environment. Same checkpoint, same 25 held-out seeds:

| Evaluation | lives saved /40 | severe save rate |
|---|---|---|
| Deterministic (argmax) | 12.68 | 0.262 |
| **Stochastic (sample)** | **17.52** | **0.332** |

A **38% difference in lives saved**, from the evaluation protocol alone.

## Why

The task is fundamentally a **search** problem. A deterministic policy maps each observation
to exactly one action, so an agent that returns to a previously seen state repeats the same
action and can cycle — pacing a corridor, re-entering the same room. Sampling breaks those
cycles and keeps the agent covering ground. Stochasticity is not merely exploration noise
here; it is part of what makes the policy work.

This is a property of partially observable search tasks generally: with a limited view, many
distinct world states produce identical observations, and a deterministic policy cannot
distinguish them.

## Consequence for earlier numbers

The first mandate sweep evaluated deterministically, so **its eval figures are understated**.
Concretely, it made PPO look roughly tied with the greedy heuristic (≈8 lives each) when
stochastic evaluation puts PPO at ~17.5 against greedy's 8.04 — better than 2×.

Any conclusion drawn from those deterministic eval numbers would have been wrong, and in a
direction that understates the method.

## What changed

- **Stochastic evaluation is now the default**; `--eval-deterministic` is opt-in.
- `experiments/reevaluate.py` reloads each run's final checkpoint and re-evaluates on
  held-out seeds, writing `eval_stochastic.json` beside the original summary. No retraining
  is needed.
- Training-time metrics were never affected — training always sampled.

## Note for the write-up

Worth stating explicitly in the paper's evaluation section, with the number. Reviewers expect
deterministic evaluation by default, so the deviation needs justifying — and the justification
is a measured 38% gap, not a preference.
