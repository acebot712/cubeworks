#!/bin/zsh
# The whole remaining experiment, as one idempotent command.
#
# Safe to run at any time, any number of times, from any state. Every step
# checks whether its output already exists and skips it, and the two long
# training phases resume from checkpoints written every 4,000 steps. Kill it
# mid-run, close the lid, pull the power: re-running loses at most a few
# minutes of the step that was in flight.
#
#   ./run_pipeline.sh              run (or resume) everything
#   ./run_pipeline.sh --status     print progress and exit
#
# Designed to be driven by launchd so it also survives a reboot; see
# ~/Library/LaunchAgents/com.cubeworks.pipeline.plist

set -u
cd "$(dirname "$0")"
PY=../.venv-mlx/bin/python
R=../eval/results
STATUS=PIPELINE_STATUS.txt
LOG=pipeline.log

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# --- progress report ------------------------------------------------------
report() {
  {
    print -r -- "CUBEWORKS pipeline: $(date)"
    print -r -- ""
    local done=0 total=0
    for k in 2 4 6 8 10 12 16 24; do
      for s in 0 1 2; do
        total=$((total+1))
        local f="ckpt_wings-k${k}_s${s}.json"
        if [[ -f $f ]] && $PY -c "import json,sys;sys.exit(0 if json.load(open('$f')).get('step',0)>=40000 else 1)" 2>/dev/null; then
          done=$((done+1))
        fi
      done
    done
    print -r -- "  E1 ladder training      ${done}/${total} runs at 40k steps"
    print -r -- "  probed profiles         $(ls $R/probeprofile-*.json 2>/dev/null | wc -l | tr -d ' ') files"
    print -r -- "  per-shell profiles      $(ls $R/profile-*.json 2>/dev/null | wc -l | tr -d ' ') files"
    print -r -- "  budget curves           $(ls $R/budget-*.json 2>/dev/null | wc -l | tr -d ' ') files"
    print -r -- "  E5 aggregate            $([[ -f $R/e5-prediction.json ]] && echo present || echo pending)"
    print -r -- ""
    print -r -- "  running: $(pgrep -f 'davi.py|sweep.py|profile' >/dev/null && echo yes || echo no)"
  } | tee "$STATUS"
}

if [[ "${1:-}" == "--status" ]]; then report; exit 0; fi

# --- single-instance lock -------------------------------------------------
# Two copies racing would corrupt checkpoints. The lock is the PID; a stale one
# from a crash or power cut is detected and cleared rather than blocking forever.
LOCK=pipeline.lock
if [[ -f $LOCK ]]; then
  old=$(cat $LOCK 2>/dev/null)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    log "already running as pid $old: exiting"
    exit 0
  fi
  log "clearing stale lock from pid ${old:-unknown}"
fi
print -r -- $$ > $LOCK
trap 'rm -f $LOCK' EXIT INT TERM

log "=== pipeline start (pid $$) ==="

# --- E1: the size ladder, 3 seeds ----------------------------------------
# sweep.py skips any run whose checkpoint already reached the target step and
# any evaluation whose JSON already exists, so this is a no-op once complete.
log "E1: ladder sweep (8 rungs x 3 seeds, resumable)"
$PY sweep.py --steps 40000 --seeds 3 --n 200 --width 100 2>&1 | tee -a "$LOG"
report

# --- probed profiles where enumeration cannot reach -----------------------
# forward ball built once and large; backward walk per query and short. The
# opposite split cost 37 s/state at k=6.
# k=8 only. k=10 was attempted and abandoned after 31 hours: its mean distance
# is ~9.9 (from the fit 0.864k+1.242) while forward 6 + back 4 reaches only 10,
# so roughly half its states would return unresolved, the deep half the profile
# exists to measure. Covering it needs a far larger ball than fits in memory,
# which is the same wall that stops enumeration, arriving one rung later.
for k in 8; do
  out=$R/probeprofile-wings-k${k}_s0.json
  if [[ -f $out ]]; then
    log "probed profile k=$k: cached"
  else
    log "probed profile k=$k"
    $PY profile_probed.py --task wings-k$k --tag _s0 \
        --forward 6 --back 4 --cap 100000000 --per-len 60 --max-len 14 2>&1 | tee -a "$LOG"
  fi
done
report

# --- per-shell profiles on every rung with exact ground truth -------------
for spec in "wings-k4 _s0" "wings-k6 _s0" "wings-k4 _s1" "wings-k6 _s1" "wings-k4 _s2" "wings-k6 _s2"; do
  set -- ${=spec}
  [[ -f ckpt_$1$2.json ]] || continue
  if [[ -f $R/profile-$1$2.json ]]; then
    log "profile $1$2: cached"
  else
    log "profile $1$2"
    $PY profiles.py --task $1 --tag $2 --pdb-k 2 --n 30000 2>&1 | tee -a "$LOG"
  fi
done

# --- E5: does the profile predict the search budget? ----------------------
log "E5: budget curves + prediction"
$PY e5_predict.py --n-budget 100 2>&1 | tee -a "$LOG"

report
log "=== pipeline complete ==="
