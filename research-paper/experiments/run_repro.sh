#!/usr/bin/env bash
# E1 reproduction (Phase 0 gate): does the fixed code still reproduce the frozen
# headline ordering (selfish > mixed > cooperative), and which cooperative variant
# matches the frozen 50k numbers?
#
# Runs seed 0 for all three schemes, PLUS a second cooperative run using the pure
# team-average variant, so we can compare both against results/run_50000_* in the
# frozen tree. Uses the same launcher underneath.
#
# Usage:
#   ./run_repro.sh                 # 50k episodes (matches frozen run length)
#   EPISODES=30000 ./run_repro.sh  # faster first look
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP="${EPISODES:-50000}"

# jobs=4: three schemes + one extra cooperative variant fit the 4 performance cores.
echo ">>> E1a: selfish / cooperative(plus_own) / mixed, seed 0"
EPISODES="$EP" SEEDS=0 SCHEMES="selfish cooperative mixed" JOBS=3 \
  COOP_VARIANT=plus_own "$HERE/run_sweep.sh"

echo ">>> E1b: cooperative(team_avg) seed 0  (paper eq 2.7 variant, for comparison)"
EPISODES="$EP" SEEDS=0 SCHEMES="cooperative" JOBS=1 \
  COOP_VARIANT=team_avg "$HERE/run_sweep.sh"

echo ">>> E1 done. Compare research-paper/runs/results/run_${EP}_*_seed0*"
echo "    against the frozen results/run_50000_* to pick the cooperative variant."
