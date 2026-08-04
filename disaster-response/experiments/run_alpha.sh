#!/usr/bin/env bash
# Phase 1d - the mandate sweep. Tests H1 (incentive-induced triage distortion).
#
# Mandate: r_i = alpha*own + (1-alpha)*team_avg.
#   alpha = 1  -> each agency credited only for its OWN rescues
#   alpha = 0  -> everyone credited with the collective total
#
# H1 predicts: as alpha rises, agencies increasingly avoid SEVERE victims (which need two
# responders and therefore force credit-sharing) and cherry-pick claimable minor ones. So
# severe_save_rate should fall with alpha even if total lives saved does not.
#
# Outputs land in disaster-response/runs/ (git-ignored).
#
# Usage:
#   DRYRUN=1 ./run_alpha.sh
#   ./run_alpha.sh                                  # 5 alphas x 3 seeds, 8k eps (~10h at 4 jobs)
#   EPISODES=4000 SEEDS="0 1" ./run_alpha.sh        # quicker first look
set -u

EPISODES="${EPISODES:-8000}"        # past the ~4-5k plateau (see probe_convergence.md)
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"
JOBS="${JOBS:-4}"
# The chosen configuration (see config_selection.md): coordinated ceiling 0.80, so triage
# binds and H1 can express itself, while the task-imposed severe/minor gap stays small.
CONFIG="${CONFIG:---grid-size 50 --num-agents 6 --num-agencies 2 --num-victims 40 --max-steps 300}"
EXTRA="${EXTRA:-}"                # e.g. EXTRA="--independent" 

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/../runs"
LOGDIR="$OUT/_launch_logs"
mkdir -p "$OUT" "$LOGDIR"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

JOBLIST=()
for a in $ALPHAS; do for s in $SEEDS; do JOBLIST+=("$a $s"); done; done
N=${#JOBLIST[@]}
# measured ~1.2 s/episode on an M1 performance core, single-threaded
wall=$(awk "BEGIN{printf \"%.1f\", ($N/$JOBS)*$EPISODES*0.74/3600}")

cat <<EOF
=== disaster mandate sweep (Phase 1d) ===
  alphas        : $ALPHAS
  seeds         : $SEEDS
  episodes/run  : $EPISODES
  config        : $CONFIG
  extra args    : ${EXTRA:-(none)}
  total runs    : $N
  parallel jobs : $JOBS
  output        : $OUT
  est. wall     : ~${wall} h
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBLIST[@]}"; do echo "  job: alpha=${j% *} seed=${j#* }"; done
  echo "(dry run - nothing executed)"; exit 0
fi

run_one() {
  local a="$1" s="$2"
  local tag="a${a}_seed${s}"
  echo "[start] $tag ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_disaster \
      --policy ppo --alpha "$a" --seed "$s" --episodes "$EPISODES" \
      --out-root "$OUT" $CONFIG $EXTRA > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[done ] $tag ($(date +%H:%M:%S))" \
                || echo "[FAIL ] $tag rc=$rc (see $LOGDIR/${tag}.log)"
}
export -f run_one
export EPISODES PY REPO LOGDIR OUT EXTRA CONFIG

cd "$REPO" || exit 1
echo "=== launching $N runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _
echo "=== sweep finished ($(date +%H:%M:%S)) ==="
