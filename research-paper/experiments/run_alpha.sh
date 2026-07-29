#!/usr/bin/env bash
# Alpha-sweep (Phase 2 centrepiece): mixed reward across a grid of alpha values.
#
# Reward is r_i = alpha*own + (1-alpha)*team_avg, so:
#   alpha=1.0  == selfish      alpha=0.0 == cooperative(team_avg)   alpha=0.5 == mixed
# Sweeping alpha turns the three discrete conditions into a continuous curve. The
# endpoints (0.0, 1.0) double as consistency checks against the selfish / cooperative runs.
#
# All output lands in git-ignored research-paper/runs/ (dirs: run_<EP>_mixed_a<ALPHA>_seed<N>).
#
# Usage:
#   ./run_alpha.sh                                   # 5 alphas x 3 seeds, 30k, JOBS=4 (~8h)
#   ALPHAS="0.25 0.75" SEEDS="0 1 2 3 4" ./run_alpha.sh   # only the missing mid-points
#   DRYRUN=1 ./run_alpha.sh
set -u

EPISODES="${EPISODES:-30000}"
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"
JOBS="${JOBS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNDIR="$HERE/../runs"
LOGDIR="$RUNDIR/_launch_logs"
mkdir -p "$RUNDIR" "$LOGDIR"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

JOBLIST=()
for a in $ALPHAS; do
  for sd in $SEEDS; do
    JOBLIST+=("$a $sd")
  done
done
NJOBS=${#JOBLIST[@]}
wall_min=$(awk "BEGIN{printf \"%.1f\", ($NJOBS/$JOBS)*$EPISODES*0.26/60}")

cat <<EOF
=== alpha-sweep plan ===
  scheme          : mixed (r = alpha*own + (1-alpha)*team_avg)
  alphas          : $ALPHAS
  seeds           : $SEEDS
  episodes/run    : $EPISODES
  total runs      : $NJOBS
  parallel jobs   : $JOBS
  output dir      : $RUNDIR
  est. wall clock : ~${wall_min} min  (rough)
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBLIST[@]}"; do echo "  job: alpha=${j% *} seed=${j#* }"; done
  echo "(dry run — nothing executed)"
  exit 0
fi

run_one() {
  local a="$1" sd="$2"
  local tag="mixed_a${a}_seed${sd}"
  echo "[start] $tag  ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_headless \
      --num-episodes "$EPISODES" \
      --reward-scheme mixed \
      --alpha "$a" \
      --seed "$sd" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[done ] $tag  ($(date +%H:%M:%S))" \
                || echo "[FAIL ] $tag  rc=$rc  (see $LOGDIR/${tag}.log)"
}
export -f run_one
export EPISODES CHECKPOINT_EVERY PY REPO LOGDIR

cd "$RUNDIR" || exit 1
echo "=== launching $NJOBS runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _
echo "=== alpha-sweep finished ($(date +%H:%M:%S)) ==="
