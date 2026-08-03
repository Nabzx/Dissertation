# Reproducibility appendix

Everything needed to reproduce every number in the paper. Paste into the paper's appendix
(after references) or release alongside the code.

## Environment

| | |
|---|---|
| Hardware | Apple M1 (2021 MacBook Pro), 8 cores (4 performance), CPU-only — **no GPU used** |
| OS | macOS 26.2 (arm64) |
| Python | 3.14.0 |
| PyTorch | 2.11.0 |
| NumPy | 2.4.4 |
| PettingZoo | 1.24.3 |
| Gymnasium | 1.2.3 |
| Matplotlib | 3.10.8 |
| Code revision | `6b55cb7` (branch `paper`) |

## Environment configuration (fixed across all conditions)

| Parameter | Value |
|---|---|
| Grid | 25 × 25, octagonal arena mask |
| Agents | 4 |
| Resources | 25, respawn 2% per depleted node per step |
| Obstacles | 45 (static after reset) |
| Observation | 5 × 5 local window, flattened to 25 ints |
| Actions | 5 (stay, up, down, left, right) |
| Episode length | 250 steps (truncation) or all resources depleted (termination) |
| Discount γ | 0.99 |

## PPO hyperparameters

| Parameter | Value |
|---|---|
| Network | 2 × 64 MLP, ReLU, separate policy & value heads |
| Parameter sharing | shared across 4 agents (ablation: independent, `--independent`) |
| Optimiser | Adam, lr 3 × 10⁻⁴ |
| GAE λ | 0.95 |
| Clip ratio ε | 0.2 |
| Epochs / update | 4 |
| Mini-batch | 64 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Max grad norm | 0.5 |
| Update cadence | end of each episode (shared buffer over 4 agents) |

## Protocol

- **Seeding.** One master `--seed` seeds Python `random`, NumPy and PyTorch weight
  initialisation. Episode seeds are a disjoint per-seed stream (`seed × num_episodes + episode`),
  so conditions see identical environment sequences for a given seed.
- **Truncation.** The 250-step limit is truncation, not termination: the GAE tail bootstraps
  from V(s_T) per agent. Only genuine resource depletion is treated as terminal.
- **Training length.** 30,000 episodes per run. Main conditions: 5 seeds (0–4).
  α-sweep and ablation: 3 seeds (0–2).
- **Metrics.** Reported over the final 100 episodes. Efficiency = collected / spawned;
  Jain fairness over per-agent counts; cooperation = efficiency × fairness (noted in-paper
  as non-independent); free-rider fraction = mean fraction of agents collecting < 0.5 × the
  per-episode mean; contribution Gini over per-agent counts.
- **Statistics.** Two-sided Welch t-tests on per-seed final metrics, with Cohen's d.
  Implemented in pure NumPy (incomplete-beta t-distribution CDF); no SciPy dependency.

## Wall-clock

~0.26 s/episode single-threaded → ~2.2 h per 30k run. Runs are executed 4-at-a-time
(one per performance core, `OMP_NUM_THREADS=1`), so 15 runs ≈ 8 h.

## Commands

```bash
# main multi-seed experiment (3 schemes x 5 seeds)
cd research-paper/experiments && ./run_sweep.sh

# alpha-sweep (5 alphas x 3 seeds)
./run_alpha.sh

# independent-policies ablation (2 schemes x 3 seeds)
./run_ablation.sh

# analysis
python aggregate.py   --episodes 30000    # table + Welch t-tests + Cohen's d
python plot_curves.py --episodes 30000    # Fig 2: learning curves w/ CI bands
python plot_alpha.py  --episodes 30000    # Fig 3: alpha dose-response curve
python freerider.py   --episodes 30000    # free-riding metrics
```

## Consistency checks performed

1. **α endpoints.** Mixed reward at α=1 is byte-identical to the selfish condition, and at
   α=0 byte-identical to cooperative(team-average), per seed — verifying the reward family
   reduces correctly (e.g. seed 0 efficiency: α=1 = selfish = 0.6552; α=0 = cooperative = 0.5308).
2. **Determinism.** A fixed seed reproduces a run exactly; different seeds diverge
   (confirming network initialisation is genuinely seeded).
3. **No regression.** Shared-weight results are unchanged after adding the independent-policy
   code path.
