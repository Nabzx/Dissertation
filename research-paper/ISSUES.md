# Issue catalogue (code ↔ paper)

Concrete problems found in the frozen dissertation code, ranked by how much they matter
for publication. Line references are to the original (frozen) tree. Fixes are applied only
to the fork (Phase 0), never to these files.

Severity: 🔴 blocks publication · 🟠 must fix for credibility · 🟡 polish / nice-to-have

---

### #1 🔴 Reward equation: paper ≠ code
- **Where:** `env/rewards.py:19` — cooperative returns `team_avg + 0.1 * own`.
- **Paper says:** eq 2.7 and §3.4.2 describe the pure team average `(1/N) Σ r_j`; the
  "avoiding hard-coded cooperation" argument assumes individual contribution is invisible.
  The `+ 0.1*own` term leaks an individual signal, which partly contradicts the framing.
- **Action:** Determine which variant produced the 50k results (re-run both, compare to
  logged `summary.json`). Then make the paper's equation and the code identical. Either is
  defensible — but they must match.

### #2 🟠 Truncation treated as termination
- **Where:** `train/run_simulation.py:183` always calls
  `ppo_agent.update(last_value=0.0, last_done=True)`; per-step `done` at line 136 is
  `terminations OR truncations`.
- **Why it matters:** Episodes almost always end by the 250-step truncation, not by
  resource depletion. Correct PPO/GAE bootstraps the final value from `V(s_T)` on
  truncation; here it's zeroed, biasing value targets near episode end. `PPOAgent.update`
  already accepts `last_value`/`last_done`, so the fix is in the caller.
- **Action:** On truncation, pass the bootstrap value estimate and `last_done=False`;
  reserve `last_done=True` for genuine terminal (all resources depleted).

### #3 🔴 Incomplete / shared seeding
- **Where:** `train/run_simulation.py:40` (`env.reset(seed=episode_num)`); PPO net init in
  `agents/ppo_agent.py:74` is unseeded.
- **Why it matters:** (a) PyTorch weight init is never seeded → each condition starts from a
  *different* random network, so the conditions aren't controlled at initialisation. (b)
  Only one run per condition. (c) The same episode-seed sequence (0..49999) is reused across
  conditions. Together this is the single biggest barrier to publication.
- **Action:** Single `--seed` controlling Python `random`, NumPy and `torch.manual_seed`;
  run multiple seeds (Phase 1).

### #4 🟠 Cooperation score is circular
- **Where:** analysis metrics (`analysis/metrics.py`, reported in §5.3.2).
- **Why it matters:** "Cooperation score" is defined as efficiency × fairness, so it cannot
  be independent evidence of cooperation — a reviewer will flag this immediately. Paper even
  concedes it in §5.5.
- **Action:** Add a genuinely independent cooperation/free-riding metric (see ROADMAP
  Phase 2).

### #5 🟡 Communication layer ignores its own sparsity flag
- **Where:** `agents/communication.py` `update_messages_after_step()` broadcasts every agent
  every step; the 2% `just_communicated` gating described in §3.5 isn't applied in training.
- **Why it matters:** Low — the headline 50k runs have `use_communication=False`
  (`train/train_headless.py:50`), so results are unaffected. Only relevant if we revive the
  communication experiment (Objective 5).
- **Action:** If we include a comms experiment, make the gating match the paper; otherwise
  drop comms from the paper scope and say so.

### #6 🟡 Stale action-space equation + weak bibliography
- **Where:** paper eq 3.2 lists `{move, attack, gather, rest, propose alliance, accept,
  defect}` — leftover from an abandoned design; only 5 movement actions exist
  (`{stay,up,down,left,right}`). Bibliography leans on Medium / ResearchGate figures.
- **Action:** Drop eq 3.2's phantom actions in the paper. Replace informal citations with
  peer-reviewed MARL sources (see `notes/related-work.md`).

---

## Not bugs — design choices to *defend* in the paper
- Shared-weight single policy across 4 agents (§5.5): reframe as "parameter-shared
  independent learners"; acknowledge it observes one policy, not 4 distinct strategies.
- 2% resource respawn, octagonal mask, 5×5 window: fine, just state them as fixed controls.
- Small 2×64 MLP: defensible (interpretability), keep the justification.
