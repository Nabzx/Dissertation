# Disaster Response — multi-agency coordination under incentive design

Branch `disaster`, forked from `paper` so it inherits the corrected PPO (global seeding,
truncation bootstrapping) and the α reward machinery. **`env/gridworld_env.py` is untouched**,
so every workshop-paper result remains reproducible.

## The idea in one line
Independent response agencies are rewarded `α·(own rescues) + (1−α)·(collective rescues)`.
How does that mandate shape whether they coordinate, specialise, and share information?

Read [DESIGN.md](DESIGN.md) for the environment design and the phased plan.

## Status
- [x] **Phase 1a** — `env/disaster_env.py` built and mechanically verified:
      severe victims require two simultaneous responders; victims die on a countdown;
      multi-channel observations (obstacle / victim / severity / urgency / own-agency /
      other-agency).
- [ ] Phase 1b — wire to PPO (obs is now 3-D, so the policy needs a flatten or small conv)
- [ ] Phase 1c — α sweep: does the free-riding effect transfer to a provision dilemma?
- [ ] Phase 2 — heterogeneous roles
- [ ] Phase 3 — memory (recurrent policy)
- [ ] Phase 4 — communication, and whether agencies learn to withhold information

## Why this environment is different from the gridworld
| | Gridworld (paper) | Disaster response |
|---|---|---|
| Dilemma type | appropriation (taking) | **provision (contributing)** |
| Cost of idling | forgone gain | **destroyed value — victims die** |
| Interdependence | none (only collision avoidance) | **severe victims need 2 responders** |
| Observation | one integer per cell | **6 channels** |
| Collective metric | resources collected | **lives saved** |

## Framing discipline
A **stylised model of multi-agency coordination** — not a disaster simulator, and never tied
to any specific real incident. Claims stay at the level of incentive structure and coordination.
