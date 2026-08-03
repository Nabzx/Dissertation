#!/usr/bin/env bash
# Unattended multi-seed training sweep for the paper (Phase 1).
#
# Runs every (reward scheme x seed) combination, in parallel, single-threaded per
# job so they pack onto the M1's performance cores. All output (checkpoints, results,
# logs) lands under research-paper/runs/ and is git-ignored automatically.
#
# Nothing outside research-paper/ is touched: this only *reads* the frozen code via
# PYTHONPATH and *writes* into research-paper/runs/.
#
# Usage (env-var config, all optional):
#   ./run_sweep.sh                                  # defaults below
#   EPISODES=50000 SEEDS="0 1 2 3 4" ./run_sweep.sh
#   JOBS=2 SCHEMES="selfish cooperative" ./run_sweep.sh
#   COOP_VARIANT=team_avg SCHEMES=cooperative SEEDS=0 ./run_sweep.sh
#   DRYRUN=1 ./run_sweep.sh                         # print the plan, run nothing
#
set -u

# --- config ---
EPISODES="${EPISODES:-30000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SCHEMES="${SCHEMES:-selfish cooperative mixed}"
JOBS="${JOBS:-4}"                 # parallel jobs; M1 has 4 performance cores
COOP_VARIANT="${COOP_VARIANT:-team_avg}"   # paper's cooperative condition (eq 2.7, alpha=0 endpoint)
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

# --- paths (script lives in research-paper/experiments/) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"              # repo root (frozen code)
RUNDIR="$HERE/../runs"                          # all outputs go here
LOGDIR="$RUNDIR/_launch_logs"
mkdir -p "$RUNDIR" "$LOGDIR"

# pick the project venv python if present
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# --- build the job list (scheme seed pairs) ---
JOBLIST=()
for s in $SCHEMES; do
  for sd in $SEEDS; do
    JOBLIST+=("$s $sd")
  done
done
NJOBS=${#JOBLIST[@]}

# --- rough time estimate (~0.26 s/episode single-thread on M1) ---
per_run_min=$(awk "BEGIN{printf \"%.1f\", $EPISODES*0.26/60}")
wall_min=$(awk "BEGIN{printf \"%.1f\", ($NJOBS/$JOBS)*$EPISODES*0.26/60}")

cat <<EOF
=== training sweep plan ===
  schemes         : $SCHEMES
  seeds           : $SEEDS
  episodes/run    : $EPISODES
  coop variant    : $COOP_VARIANT
  total runs      : $NJOBS
  parallel jobs   : $JOBS
  output dir      : $RUNDIR
  python          : $PY
  est. per run    : ~${per_run_min} min
  est. wall clock : ~${wall_min} min  (rough)
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  printf '  job: %s\n' "${JOBLIST[@]}"
  echo "(dry run — nothing executed)"
  exit 0
fi

# --- one run ---
run_one() {
  local scheme="$1" seed="$2"
  local tag="${scheme}_seed${seed}_${COOP_VARIANT}"
  echo "[start] $tag  ($(date +%H:%M:%S))"
  # single-threaded so parallel jobs don't fight over cores
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_headless \
      --num-episodes "$EPISODES" \
      --reward-scheme "$scheme" \
      --seed "$seed" \
      --cooperative-variant "$COOP_VARIANT" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[done ] $tag  ($(date +%H:%M:%S))"
  else
    echo "[FAIL ] $tag  rc=$rc  (see $LOGDIR/${tag}.log)"
  fi
}
export -f run_one
export EPISODES COOP_VARIANT CHECKPOINT_EVERY PY REPO LOGDIR

# --- run in research-paper/runs so all relative outputs land there ---
cd "$RUNDIR" || exit 1
echo "=== launching $NJOBS runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _

echo "=== sweep finished ($(date +%H:%M:%S)) ==="
echo "results under: $RUNDIR/results/   logs under: $LOGDIR/"
