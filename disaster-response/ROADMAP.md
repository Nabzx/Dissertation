# Disaster Response — goal, premise, and plan

## The premise

When several independent organisations respond to the same crisis, each is measured largely on
**its own** outcomes — its rescues, its beneficiaries, the numbers it reports to its funders.
But the outcome that matters is the **collective** one: total lives saved. Effort is costly and
credit is individual, so each agency faces a provision dilemma: contribute effort that may
benefit a rival's numbers, or concentrate on what it can claim itself.

This is a documented failure mode in humanitarian response — duplicated effort, unshared
information, and competition for visibility between agencies.

We model this as a controlled learning problem. Independent PPO responders, grouped into
agencies, are rewarded

    r_i = α · (own rescues) + (1 − α) · (collective rescues)

and α — the **mandate design** — is the single independent variable. Everything else is held
fixed, exactly as in the companion gridworld study.

## The research question

> **What actually makes independent response agencies coordinate better — and by how much?**

Framed as **diagnosis → cure**, not state-of-the-art chasing. We first establish a coordination
failure caused by the credit structure, then test a ladder of interventions and report which
ones improve lives saved. Each intervention is compared against *the same system without it*,
which is a fair and winnable comparison — we are not claiming to beat the MARL literature.

The underlying variable remains the mandate design α.

Four sub-questions, one per phase:

1. Does under-contribution emerge in a *provision* dilemma, and how does it scale with the
   number of agencies?
2. Does role specialisation emerge, or must it be imposed?
3. Does memory (recurrence) change coordination?
4. Do agencies learn to **withhold information** from one another?

## The two headline results we are hunting

### H1 — Incentive-induced triage distortion *(reachable in Phase 1)*
Severe victims require **two** responders acting together; minor victims require one. Under a
strongly individual mandate (high α), a responder who joins a severe rescue must share the
credit, while a minor rescue is claimable alone.

**Prediction:** as α rises, agencies increasingly neglect victims who need cooperation.
Severe-victim save rate falls even where total rescues stay flat or rise.

If it holds, the claim is: *individually-credited mandates cause agencies to prefer easy,
claimable cases over hard ones that require cooperation* — a distortion produced entirely by
the incentive, not by any programmed preference. This mirrors a real criticism of outcome-count
metrics in aid work.

### H2 — Emergent information withholding *(Phase 4)*
Agents can broadcast victim sightings. Sharing a sighting helps whoever reaches it first —
possibly a rival agency.

**Prediction:** sharing rate falls as α rises; collective lives saved falls with it.

Claim: *self-interested agencies learn to conceal information, with no deception programmed in.*

Both are falsifiable, and a negative result on either is still informative.

## Why this is not a re-run of the gridworld paper

| | Gridworld | Disaster response |
|---|---|---|
| Dilemma | appropriation (taking) | **provision (contributing)** |
| Idling | forgoes gain | **destroys value — victims die** |
| Interdependence | none | **severe victims need 2 responders** |
| New phenomena | — | triage distortion, information withholding |

In experimental economics, provision and appropriation dilemmas produce *different* human
behaviour, so whether the effect transfers is a genuine question.

---

## Plan

Each phase must yield a result before the next begins. If a phase produces nothing clean,
stop and diagnose rather than stacking another subsystem on top.

### Phase 1 — Core environment and the α sweep
**1a ✅ done** — environment built, mechanics verified (severe victims need two responders;
victims die on a countdown; multi-channel observations).

**1b — wire to PPO.** Observations are 6×7×7, so the policy needs an encoder. Start with a
flatten (294 → MLP) for speed; test a small conv later as an ablation.

**1c — baselines.** Three rungs: random (floor), greedy (decentralised, uncoordinated), and
**coordinated** (privileged central planner with BFS — the ceiling, bounding what coordination
could achieve). Plus deterministic evaluation on held-out seeds, so final numbers are not
contaminated by training-time exploration noise.

**1d — the α sweep.** α ∈ {0, 0.25, 0.5, 0.75, 1} × 3 seeds.
*Outcomes:* lives saved, **severe vs minor save rate (tests H1)**, effort share, idle rate.

**1e — agency-count sweep.** Agencies ∈ {1, 2, 4, 8} at fixed responder count — does
coordination degrade as more organisations are involved (the 1/N result, in a new setting)?

### Phase 2 — Roles
Heterogeneous capabilities (medic / transport / scout). Question: does specialisation emerge
from identical agents, or must it be assigned? Compare emergent vs imposed roles.

### Phase 3 — Memory
Recurrent policy (GRU). Question: does remembering victim locations change coordination, and
does it change the α response? Justified because victims seen but not yet reached are exactly
what an agent must carry across time.

### Phase 4 — Communication *(the headline)*
A broadcast channel carrying victim sightings. Measure sharing rate, honesty, and collective
outcome against α. Tests **H2**.

### Phase 4b — Robustness / safety
What happens when an agency **fails or withdraws** mid-response, or behaves purely selfishly
while others cooperate? Does the system degrade gracefully or collapse? This is the safety
dimension: a coordination scheme that only works when everyone behaves is not much use.

### The intervention ladder (the paper's spine)
Each rung is measured on **lives saved** and **severe-victim save rate**, against the same
system without it:

| Intervention | Question |
|---|---|
| Mandate redesign (α) | how much collective credit is needed? |
| Communication | does sharing sightings help, and is it used? |
| Role assignment | does specialisation help, emergent or imposed? |
| Memory (recurrence) | does remembering victims help? |
| Task claiming | does explicit allocation beat implicit? |

Honest reporting includes the rungs that **don't** work.

### Phase 5 — Write-up
Target the same venues as the gridworld paper (ALA, EWRL, Cooperative AI, ICBINB), or a
combined provision-vs-appropriation paper for TMLR.

---

## Metrics

| Metric | Measures |
|---|---|
| Lives saved / lost | collective outcome |
| **Severe vs minor save rate** | **H1 — triage distortion** |
| Joint-rescue count | actual cooperation events |
| Effort share, Jain over agencies | contribution inequality |
| Idle rate | free-riding (proved highly informative in the gridworld) |
| **Sharing rate, honesty** | **H2 — information withholding** |

## Risks
- **Compute.** Bigger grid, more agents, recurrence → expect 3–5× the gridworld's cost per run.
  Keep the grid moderate and episodes tight, or sweeps become unaffordable on an M1.
- **Scope.** Four subsystems. One result per phase, or this becomes a demo.
- **Learnability.** The task is harder than foraging; if PPO cannot learn it at all, simplify
  (fewer obstacles, larger view, shorter episodes) before adding anything.
- **Framing.** A stylised model of coordination — never a disaster simulator, never tied to a
  specific real incident.
