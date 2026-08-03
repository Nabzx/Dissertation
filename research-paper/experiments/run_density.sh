#!/usr/bin/env bash
# Resource-density sweep: does the selfish-vs-cooperative effect generalise beyond the
# default 25-resource configuration?
#
# Scarcity is the pressure that makes the social dilemma bite, so varying resource count
# tests whether the result is specific to one level of competition. Runs land in
# run_<EP>_<scheme>_seed<N>[_team_avg]_r<RES> (the _r suffix is omitted at the default 25,
# so existing runs keep their names and are reused as the mid-point).
#
# NOTE: efficiency is normalised by the run's own resource count (post_training_analysis
# receives max_resources), so densities are directly comparable.
#
# Usage:
#   ./run_density.sh                              # 15 & 40 resources x selfish/cooperative x 3 seeds
#   DENSITIES="10 15 40" SEEDS="0 1 2" ./run_density.sh
#   DRYRUN=1 ./run_density.sh
set -u

EPISODES="${EPISODES:-30000}"
SEEDS="${SEEDS:-0 1 2}"
SCHEMES="${SCHEMES:-selfish cooperative}"
DENSITIES="${DENSITIES:-15 40}"   # 25 (default) already covered by the main sweep
JOBS="${JOBS:-4}"
COOP_VARIANT="${COOP_VARIANT:-team_avg}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNDIR="$HERE/../runs"
LOGDIR="$RUNDIR/_launch_logs"
mkdir -p "$RUNDIR" "$LOGDIR"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

JOBLIST=()
for r in $DENSITIES; do
  for s in $SCHEMES; do
    for sd in $SEEDS; do
      JOBLIST+=("$r $s $sd")
    done
  done
done
NJOBS=${#JOBLIST[@]}
wall_min=$(awk "BEGIN{printf \"%.1f\", ($NJOBS/$JOBS)*$EPISODES*0.26/60}")

cat <<EOF
=== resource-density sweep plan ===
  densities       : $DENSITIES  (25 = default, already run)
  schemes         : $SCHEMES
  seeds           : $SEEDS
  episodes/run    : $EPISODES
  total runs      : $NJOBS
  parallel jobs   : $JOBS
  est. wall clock : ~${wall_min} min  (rough)
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBLIST[@]}"; do
    set -- $j; echo "  job: resources=$1 scheme=$2 seed=$3"
  done
  echo "(dry run — nothing executed)"
  exit 0
fi

run_one() {
  local res="$1" scheme="$2" sd="$3"
  local tag="${scheme}_r${res}_seed${sd}"
  echo "[start] $tag  ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_headless \
      --num-episodes "$EPISODES" \
      --reward-scheme "$scheme" \
      --num-resources "$res" \
      --seed "$sd" \
      --cooperative-variant "$COOP_VARIANT" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[done ] $tag  ($(date +%H:%M:%S))" \
                || echo "[FAIL ] $tag  rc=$rc  (see $LOGDIR/${tag}.log)"
}
export -f run_one
export EPISODES COOP_VARIANT CHECKPOINT_EVERY PY REPO LOGDIR

cd "$RUNDIR" || exit 1
echo "=== launching $NJOBS runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 3 bash -c 'run_one "$@"' _
echo "=== density sweep finished ($(date +%H:%M:%S)) ==="
echo "analyse: aggregate.py --episodes $EPISODES  (density dirs carry an _r<N> suffix)"
