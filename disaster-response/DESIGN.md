# Multi-agency disaster response — environment design

Branch: `disaster` (off `paper`, so it inherits seeded PPO, truncation bootstrapping and the
α reward machinery). The original `gridworld_env.py` is **untouched** — this is a new
environment in `env/disaster_env.py`, so every result in the workshop paper stays reproducible.

## The research question

Same independent variable as the workshop paper, new dilemma type:

> Agencies are rewarded `r_i = α·(own rescues) + (1−α)·(collective rescues)`.
> How does the mandate design α shape coordination between independent responders?

**Why this is a genuinely different setting.** The gridworld is an *appropriation* dilemma
(agents take from a shared pool; failure = over-extraction). This is a *provision* dilemma
(agents expend effort for collective benefit; failure = under-contribution). In experimental
economics these produce different human behaviour — so whether the effect transfers is a real
question, not a re-run.

## Core design decisions

### 1. Genuine interdependence (the key difference)
In the gridworld agents never *need* each other — cooperation is only ever about staying out
of the way. Here, **severe victims require two responders simultaneously** to be rescued.
That creates a real coordination requirement, not merely parallel work. This single mechanic
is what makes the environment a coordination problem rather than a foraging problem.

### 2. Inaction is costly
Victims carry a countdown; when it expires they die and are removed. Doing nothing therefore
destroys value, unlike the gridworld where idling merely forgoes it. This makes free-riding
*visibly* harmful and gives a natural collective metric (lives lost).

### 3. Multi-channel observations (replaces integer cell codes)
The gridworld encoded everything as a single integer per cell (agent=2..N+1, obstacle=N+2),
which cannot express severity, urgency or agency membership. Here each agent sees a
`C × W × W` local window with separate channels:

| channel | contents |
|---|---|
| 0 | obstacle / rubble |
| 1 | victim present |
| 2 | victim severity (normalised) |
| 3 | victim urgency (time-to-death, normalised) |
| 4 | responders from **own** agency |
| 5 | responders from **other** agencies |

Cleaner, extensible, and it removes the value-collision problem entirely.

### 4. Agencies
`num_agents` responders partitioned into `num_agencies` groups. α is applied *within agency*:
an agency's reward blends its own rescues with the global total. With `num_agencies = num_agents`
this reduces to the fully individual case; with `num_agencies = 1` to the fully collective one.

### 5. Action space
`{stay, up, down, left, right, rescue}` — 6 actions. `rescue` only has effect when co-located
with a victim; severe victims need two responders issuing `rescue` on the same cell in the
same timestep.

## Metrics
- **Lives saved** / lives lost (primary collective outcome)
- **Effort share per agency** (the provision analogue of the free-rider fraction)
- **Idle rate** (the passivity measure that proved so informative in the gridworld)
- **Severe-rescue rate** — how often the 2-agent coordination actually happens
- Jain fairness over per-agency contribution

## Phased plan (a result per phase, no feature without a question)

| Phase | Adds | Question |
|---|---|---|
| **1** | core env + α sweep | Does the free-riding effect transfer to a provision dilemma? |
| 2 | heterogeneous roles | Does specialisation emerge, or must it be assigned? |
| 3 | recurrent policy (memory) | Does remembering victim locations change coordination? |
| 4 | communication channel | **Do agencies learn to withhold information when rewarded individually?** |

Phase 4 is the headline target: strategic information withholding emerging from incentives
alone, with no deception programmed in.

## Framing discipline
This is a **stylised model of multi-agency coordination**, never a disaster simulator, and it
must not be attached to any specific real incident. Claims stay at the level of incentive
structure and coordination, which is what the model actually supports.
