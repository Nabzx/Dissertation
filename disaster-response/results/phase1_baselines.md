# Phase 1c — baselines and learnability check

Establishes the floor and a competent-but-uncoordinated reference, so that any trained
policy's "lives saved" can be interpreted. 30 victims per episode, 8 responders, 2 agencies,
48×48 village, 400 steps.

## Results (60 episodes each, seed 0)

| Policy | lives saved /30 | save rate | **severe** save rate | **minor** save rate | joint rescues | idle rate |
|---|---|---|---|---|---|---|
| Random | 1.25 | 0.042 | **0.000** | 0.069 | 0.00 | 0.167 |
| Greedy (uncoordinated) | 4.63 | 0.154 | **0.060** | 0.219 | 0.68 | 0.000 |
| PPO after only 200 eps | 4.34 | 0.145 | 0.043 | 0.210 | 0.49 | 0.113 |

## What this tells us

1. **The task is learnable.** PPO matches the greedy heuristic after just 200 episodes,
   starting from 0–1 rescues. This was the main technical risk and it is cleared.
2. **Random never coordinates.** Zero severe rescues and zero joint rescues across 60
   episodes — severe victims are genuinely unreachable without deliberate cooperation, so the
   mechanic works as designed.
3. **The greedy reference already shows the gap H1 is about.** Greedy saves minor victims at
   0.219 but severe at 0.060 — a **3.7× gap** — purely because each agent independently chases
   its own nearest victim and severe rescues happen only by accident. This quantifies "what
   you get with zero coordination," and it is the gap a well-designed mandate should close.
4. **Severe-victim save rate is the right dependent variable.** It separates competence
   (finding victims) from coordination (rendezvousing), which total lives saved conflates.

## Timing
~1.2 s/episode single-threaded on an M1 performance core → an 8,000-episode run ≈ 2.7 h;
the 5α × 3-seed sweep ≈ 10 h at 4 concurrent jobs (one overnight).

Baselines will be re-run with more episodes for the final write-up; these numbers are for
calibration and the learnability gate.
