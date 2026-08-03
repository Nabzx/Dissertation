# Choosing the experimental configuration

The environment configuration determines whether coordination matters at all, so it must be
chosen deliberately and then frozen across every condition.

## Two criteria

1. **Triage must bind.** If a perfect coordinator can save everyone, there is no prioritisation
   dilemma and H1 (triage distortion) cannot express itself. We want a ceiling clearly below
   100%.
2. **The ceiling's severe/minor gap must be small.** If even perfect coordination saves far
   fewer severe victims than minor ones, then any gap we observe in learned policies is
   attributable to *task structure*, not incentives. We need the task-imposed gap to be small
   so that an incentive-induced gap is identifiable.

## Coordinated-ceiling measurements (40 episodes each)

| Config | grid | agents | victims | steps | ceiling | severe | minor | gap |
|---|---|---|---|---|---|---|---|---|
| base | 40 | 6 | 20 | 300 | 0.99 | 1.00 | 0.99 | −0.01 |
| **harder** | **50** | **6** | **40** | **300** | **0.80** | **0.77** | **0.82** | **0.05** |
| tight | 50 | 6 | 40 | 200 | 0.52 | 0.49 | 0.55 | 0.06 |
| scarce | 60 | 6 | 50 | 250 | 0.49 | 0.47 | 0.50 | 0.03 |

## Decision: `harder` (50×50, 6 responders, 40 victims, 300 steps)

- Ceiling of 0.80 — **triage binds** (a fifth cannot be saved even with perfect coordination)
- Task-imposed severe/minor gap is only 0.05, so a larger learned gap is attributable to the
  mandate rather than to the task
- Leaves substantial headroom (0.80) for learning to demonstrate improvement, unlike `tight`
  and `scarce` where roughly half the victims are unreachable regardless

`base` is rejected: at a 0.99 ceiling nobody has to choose whom to save, so the central
phenomenon cannot appear.

## The baseline ladder at the chosen config
To be re-measured at `harder` before the sweep, but the ladder shape from `base` was:

| Policy | interpretation |
|---|---|
| Random | floor — never coordinates |
| Greedy (decentralised, uncoordinated) | competence without coordination |
| **Coordinated (privileged, BFS, central assignment)** | **ceiling — what coordination could achieve** |

The coordinated policy is deliberately privileged (full map, central assignment) and is not
claimed to be a fair comparison. Its role is to bound what is achievable, so that the gap
between greedy and the ceiling can be read as genuine coordination headroom rather than a
limit imposed by the task.
