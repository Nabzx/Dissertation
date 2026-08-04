#!/usr/bin/env bash
# Phase 4 - the communication sweep. Tests H2 (emergent information withholding).
#
# Agents gain a broadcast head: each step they choose, independently of movement, whether to
# publish the most urgent victim they can see to a shared board that every agency can read.
# Broadcasting costs no movement, so declining to share carries no opportunity cost - any
# reduction in sharing is attributable to the mandate rather than to being busy.
#
# H2 predicts: as alpha rises (agencies credited only for their own rescues), sharing_rate
# falls, because a shared sighting may be rescued by a rival agency. Collective lives saved
# should fall with it.
#
# Compare against the Phase 1d runs at the same alpha WITHOUT communication to see whether the
# channel helps at all, and by how much it varies with the mandate.
#
# Usage:
#   DRYRUN=1 ./run_comm.sh
#   ./run_comm.sh                            # 3 alphas x 3 seeds
#   ALPHAS="0.0 1.0" SEEDS="0 1 2" ./run_comm.sh
set -u

EPISODES="${EPISODES:-8000}"
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.0 0.5 1.0}"
JOBS="${JOBS:-4}"
CONFIG="${CONFIG:---grid-size 50 --num-agents 6 --num-agencies 2 --num-victims 40 --max-steps 300}"

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
wall=$(awk "BEGIN{printf \"%.1f\", ($N/$JOBS)*$EPISODES*0.8/3600}")

cat <<EOF
=== disaster communication sweep (Phase 4) ===
  alphas        : $ALPHAS
  seeds         : $SEEDS
  episodes/run  : $EPISODES
  config        : $CONFIG
  total runs    : $N
  parallel jobs : $JOBS
  est. wall     : ~${wall} h
EOF

if [ "${DRYRUN:-0}" = "1" ]; then
  for j in "${JOBLIST[@]}"; do echo "  job: alpha=${j% *} seed=${j#* } (comm)"; done
  echo "(dry run - nothing executed)"; exit 0
fi

run_one() {
  local a="$1" s="$2"
  local tag="comm_a${a}_seed${s}"
  echo "[start] $tag ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$REPO" "$PY" -m train.train_disaster \
      --policy ppo --communication --alpha "$a" --seed "$s" --episodes "$EPISODES" \
      --out-root "$OUT" $CONFIG > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[done ] $tag ($(date +%H:%M:%S))" \
                || echo "[FAIL ] $tag rc=$rc (see $LOGDIR/${tag}.log)"
}
export -f run_one
export EPISODES PY REPO LOGDIR OUT CONFIG

cd "$REPO" || exit 1
echo "=== launching $N runs, $JOBS at a time ==="
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _
echo "=== comm sweep finished ($(date +%H:%M:%S)) ==="
