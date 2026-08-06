# Phase 1d — the mandate sweep: H1 is not supported

15 runs (5 α × 3 seeds), 8000 episodes, 50×50 / 6 responders / 2 agencies / 40 victims /
300 steps. Evaluated on 60 held-out episodes with **stochastic** action selection
(see `eval_mode.md` — argmax understates performance by ~40% here).

## Results

| α | lives /40 | severe rate | minor rate | severe/minor | joint rescues |
|---|---|---|---|---|---|
| 0.00 (collective) | 10.86 ± 0.73 | 0.198 ± 0.019 | 0.320 ± 0.028 | 0.619 | 3.19 |
| 0.25 | 12.84 ± 1.07 | 0.225 ± 0.021 | 0.385 ± 0.032 | 0.584 | 3.62 |
| 0.50 | 12.97 ± 1.56 | 0.220 ± 0.034 | 0.395 ± 0.048 | 0.557 | 3.57 |
| 0.75 | 13.34 ± 1.56 | 0.230 ± 0.013 | 0.404 ± 0.055 | 0.569 | 3.68 |
| 1.00 (agency) | 13.20 ± 1.09 | 0.215 ± 0.028 | 0.407 ± 0.035 | 0.528 | 3.45 |

Reference: random 0.77 · greedy 8.04 · **coordinated ceiling 32.59**.

### Significance, α=0 vs α=1 (Welch)

| Metric | diff | p | Cohen's d |
|---|---|---|---|
| Lives saved | −2.34 | **0.043 \*** | −2.53 |
| Minor save rate | −0.087 | **0.030 \*** | −2.76 |
| Severe save rate | −0.017 | 0.43 (n.s.) | −0.73 |
| Joint rescues | −0.26 | 0.58 (n.s.) | −0.50 |

## H1 is not supported

H1 predicted that as α rises, agencies would **avoid** victims requiring cooperation:
severe save rate and joint rescues should fall. They do not. Both are statistically flat
across the whole mandate range (p = 0.43 and 0.58).

The triage-distortion story, as stated, is wrong in this environment. Recording it plainly
rather than reframing it after the fact.

## What the data does show

**1. The gridworld result replicates in a provision dilemma.** Collective reward (α=0) is
significantly worse overall (−2.34 lives, p=0.043, d=2.5), with the same shape as the
appropriation-dilemma study: worst at α=0, a jump by α=0.25, then a plateau. Given that
provision and appropriation dilemmas produce different behaviour in humans, the transfer of
the credit-assignment effect across dilemma types is itself a result.

**2. The benefit of individual incentive accrues entirely to work that does *not* require
cooperation.** This is the interesting part. Minor rescues (one responder) improve
significantly with α (+0.087, p=0.030, d=2.8). Severe rescues (two responders) do not move
at all. So the severe/minor ratio declines steadily (0.619 → 0.528) — not because cooperative
rescues get worse, but because only the solo work gets better.

Put plainly: **sharpening individual incentives improves the easy half of the job and leaves
the part that needs teamwork untouched.** A mandate designer reading only total lives saved
would conclude the incentive worked; disaggregating by whether the task needs cooperation
shows the improvement is one-sided.

**3. Large unrealised headroom.** PPO reaches ~13/40 against a coordinated ceiling of 32.6 —
about 40%. Coordination, not competence, is the binding constraint, which is what makes the
intervention ladder (communication, roles, memory) worth testing.

## Consequences for the plan
- H1 is closed as unsupported. The paper should report it as a falsified prediction.
- The replacement claim — incentive gains are confined to non-cooperative work — is supported
  and needs a targeted test: does it hold when cooperation is made *easier* (roles/medics) or
  *better informed* (communication)?
- H2 (information withholding) is untouched by this and remains the headline candidate.
