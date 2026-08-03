#!/usr/bin/env bash
# Independent-policies ablation: does the headline effect survive without parameter sharing?
#
# The main experiments share one policy across all four agents, so a reviewer can argue the
# "four agents" are really one policy generalising over four starting positions. This runs
# selfish vs cooperative with FOUR INDEPENDENT NETWORKS (--independent). If the effect holds,
# the biggest threat to validity is answered.
#
# Output dirs: run_<EP>_<scheme>_seed<N>[_team_avg]_indep  (never collide with shared runs).
#
# Usage:
#   ./run_ablation.sh                      # 2 schemes x 3 seeds, 30k, JOBS=4 (~5h)
#   SEEDS="0 1 2 3 4" ./run_ablation.sh
#   DRYRUN=1 ./run_ablation.sh
set -u

EPISODES="${EPISODES:-30000}"
SEEDS="${SEEDS:-0 1 2}"
SCHEMES="${SCHEMES:-selfish cooperative}"
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
for s in $SCHEMES; do for sd in $SEEDS; do JOBLIST+=("$s $sd"); done; done
NJOBS=${#JOBLIST[@]}
wall_min=$(awk "BEGIN{printf \"%.1f\", ($NJOBS/$JOBS)*$EPISODES*0.30/60}")

cat <<EOF
=== independent-policies ablation plan ===
  architecture    : INDEPENDENT (one network per agent)
  schemes         : $SCHEMES
  seeds           : $SEEDS
  episodes/run    : $EPISODES
  total runs      : $NJOBS
  parallel jobs   : $JOBS
  est. wall clock : ~${wall_min} min  (rough; independent nets are a bit slower)
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBLIST[@]}"; do echo "  job: scheme=${j% *} seed=${j#* }"; done
  echo "(dry run — nothing executed)"; exit 0
fi

run_one() {
  local scheme="$1" sd="$2"
  local tag="${scheme}_seed${sd}_indep"
  echo "[start] $tag  ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_headless \
      --num-episodes "$EPISODES" --reward-scheme "$scheme" --seed "$sd" \
      --cooperative-variant "$COOP_VARIANT" --checkpoint-every "$CHECKPOINT_EVERY" \
      --independent > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[done ] $tag  ($(date +%H:%M:%S))" \
                || echo "[FAIL ] $tag  rc=$rc  (see $LOGDIR/${tag}.log)"
}
export -f run_one
export EPISODES COOP_VARIANT CHECKPOINT_EVERY PY REPO LOGDIR

cd "$RUNDIR" || exit 1
echo "=== launching $NJOBS runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _
echo "=== ablation finished ($(date +%H:%M:%S)) ==="
echo "analyse with: aggregate.py --episodes $EPISODES  (indep dirs are suffixed _indep)"
