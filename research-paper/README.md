# Research Paper Workspace

Dedicated workspace for turning the CO2304 dissertation
*"Cooperative Capabilities of Multi-Agent Systems: An AI Hunger Games Simulation"*
(First Class, 76%) into a submittable ~8-page research paper for a workshop/conference.

**Author:** Syed Nabil Shah · **Supervisor:** Dr Hammad Afzal · Advice on publication: Prof Ashiq Anjum
**Started:** 2026-07-28

---

## The one rule

**Nothing outside this `research-paper/` folder is edited or deleted.** The original
dissertation code, checkpoints, results and PDF are frozen as the submitted artifact.

When we need to change code (reward fix, seeding, new experiments), we work on a
**copy or a git branch** — see `ROADMAP.md` Phase 0. All new code, configs, runs,
figures and paper drafts live in here.

## What's in here

| File / dir | Purpose |
|---|---|
| `ROADMAP.md` | The plan — phased path from dissertation to submission, with milestones. |
| `ISSUES.md` | Catalogue of concrete code/paper issues to fix, with `file:line` references and severity. |
| `paper/OUTLINE.md` | Target venue + 8-page paper structure, section by section. |
| `experiments/` | Specs and configs for the new (multi-seed, sweep) experiments. |
| `notes/` | Literature to cite, venue notes, decisions log. |

## The core result we are selling

In a 4-agent partially-observable survival gridworld, a **selfish** reward produced
*higher* efficiency, Jain fairness and cooperation score than an explicitly
**cooperative** (team-average) reward, which collapsed early and recovered slowly.
Framed against sequential social dilemmas and the credit-assignment problem, this is a
tellable, counterintuitive story — **but it is currently supported by a single run per
condition**, which is the first thing we must fix.
