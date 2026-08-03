# Agent / PPO improvements and an honest A/B

## What was added

| Improvement | Why | Where |
|---|---|---|
| **Self search-trace channel** | Agents had no memory of where they had looked, so they re-searched cleared rooms endlessly. A decaying per-agent trace ("I have been here") is recalled even when the cell is no longer visible. | `disaster_env.py` |
| **Global position channels** | Agents knew only what was in their 9×9 window, not where they were on the map — making systematic area coverage impossible. | `disaster_env.py` |
| **Conv encoder** | The observation is spatial; flattening into an MLP discards that structure. Subclasses `PPOAgent`, so all corrected PPO machinery is inherited and conv-vs-MLP stays a clean ablation. | `conv_ppo.py` |
| **Entropy annealing** | Search needs wide exploration early and commitment later. Linear 0.03 → 0.005. | `train_disaster.py` |
| **Line of sight** (earlier) | Without it, indoor victims were visible through walls and no search was needed. | `disaster_los.py` |

All are applied identically across α conditions, so the mandate comparison stays controlled.

## A/B result (400 episodes, 40×40, 6 responders, 20 victims)

| Policy | lives saved | severe rate | minor rate | joint rescues |
|---|---|---|---|---|
| **Greedy heuristic** | **6.08** | **0.532** | 0.743 | 1.70 |
| PPO (MLP) | 4.19 | 0.290 | 0.637 | 0.95 |
| PPO (conv, cut short at 175 eps) | — | — | — | — |

## The honest read

**PPO does not beat the greedy heuristic at 400 episodes.** That is expected — greedy starts
with a hand-coded competence that PPO must discover from scratch — but it sets the bar
clearly:

1. **The runs must be far longer.** 400 episodes is early training, not converged behaviour.
   Convergence has to be demonstrated, not assumed.
2. **Greedy is the bar to beat.** A paper claiming coordination benefits must show PPO
   exceeding a competent uncoordinated heuristic; otherwise the "coordination" story is not
   supported.
3. **Conv vs MLP is unresolved.** The conv run was cut off at 175 episodes and was slightly
   behind MLP there, which is unsurprising for a larger network early in training. This needs
   a full-length comparison before choosing an encoder.

Note the severe save rates here (0.53 for greedy) are far higher than in the earlier 60×60
configuration (0.06), simply because responders are denser relative to victims, so accidental
rendezvous is common. **Configuration strongly affects how much coordination is needed** — the
final configuration must be chosen so that coordination genuinely matters, and then held
fixed across all conditions.
