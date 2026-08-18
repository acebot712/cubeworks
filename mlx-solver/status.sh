#!/bin/zsh
# What is running, how far along, and how much longer, with no Claude involved.
#
#   ./status.sh          one snapshot
#   ./status.sh -w       refresh every 15s until you stop it
#
# Reads only files the pipeline already writes, so it is safe to run at any time
# and cannot disturb a job.

cd "$(dirname "$0")"
PY=../.venv-mlx/bin/python
R=../eval/results

snapshot() {
  print -r -- "CUBEWORKS pipeline   $(date '+%a %d %b %H:%M:%S')"
  print -r -- ""

  # --- is anything alive, and is it protected from a reboot? --------------
  local agent proc
  agent=$(launchctl list 2>/dev/null | grep -c cubeworks)
  proc=$(pgrep -f "run_pipeline.sh" 2>/dev/null | head -1)
  if [[ -n "$proc" ]]; then
    print -r -- "  pipeline   RUNNING (pid $proc, up $(ps -o etime= -p $proc 2>/dev/null | tr -d ' '))"
  elif grep -q "=== pipeline complete ===" pipeline.log 2>/dev/null; then
    print -r -- "  pipeline   COMPLETE"
  else
    print -r -- "  pipeline   NOT RUNNING"
  fi
  if [[ "$agent" -gt 0 ]]; then
    print -r -- "  launchd    armed: survives reboot, logout and closing Claude"
  else
    print -r -- "  launchd    ** NOT ARMED, a reboot will not resume this **"
  fi
  print -r -- ""

  # --- current step, with its own ETA where the job reports one ----------
  local job
  job=$(pgrep -fl "davi.py|profile_probed|profiles.py|budget_curve|e5_predict|sweep.py" 2>/dev/null \
        | head -1 | sed 's/.*Python //;s/ --tag/ /' | cut -c1-70)
  [[ -n "$job" ]] && print -r -- "  now:  $job"

  # a job's own ETA line is the honest one, it knows its remaining work
  local eta
  eta=$(grep -E "ETA" pipeline.log 2>/dev/null | tail -1)
  [[ -n "$eta" ]] && print -r -- "  eta:  $(print -r -- $eta | sed 's/^ *//')"
  print -r -- ""

  # --- artefacts ---------------------------------------------------------
  local done_runs=0
  for k in 2 4 6 8 10 12 16 24; do for s in 0 1 2; do
    [[ -f ckpt_wings-k${k}_s${s}.json ]] && \
      $PY -c "import json,sys;sys.exit(0 if json.load(open('ckpt_wings-k${k}_s${s}.json')).get('step',0)>=40000 else 1)" 2>/dev/null \
      && done_runs=$((done_runs+1))
  done; done
  printf "  %-26s %s\n" "ladder training"  "${done_runs}/24 runs"
  printf "  %-26s %s\n" "probed profiles"  "$(ls $R/probeprofile-*.json 2>/dev/null | wc -l | tr -d ' ')/2"
  printf "  %-26s %s\n" "per-shell profiles" "$(ls $R/profile-*.json 2>/dev/null | wc -l | tr -d ' ')"
  printf "  %-26s %s\n" "budget curves"    "$(ls $R/budget-*.json 2>/dev/null | wc -l | tr -d ' ')"
  printf "  %-26s %s\n" "E5 aggregate"     "$([[ -f $R/e5-prediction.json ]] && date -r $R/e5-prediction.json '+%d %b %H:%M' || echo pending)"
  print -r -- ""

  # --- anything gone wrong ----------------------------------------------
  local err
  err=$(grep -E "Traceback|MemoryError|Killed|OOM|MISMATCH|disagrees" launchd.err.log pipeline.log 2>/dev/null | tail -2)
  if [[ -n "$err" ]]; then
    print -r -- "  ERRORS:"
    print -r -- "$err" | sed 's/^/    /'
  fi

  print -r -- "  last log:  $(tail -1 pipeline.log 2>/dev/null | cut -c1-72)"
}

if [[ "${1:-}" == "-w" ]]; then
  while true; do clear; snapshot; sleep 15; done
else
  snapshot
fi
