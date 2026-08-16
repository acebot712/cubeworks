"""The ladder experiment: how much compute does a learned heuristic need, as a
function of how big the problem is?

Everything except state-space size is pinned. Same 63 moves, same 600-wide
encoding, same network, same batch, same optimiser, same curriculum policy, same
evaluation protocol. Rung k=2 has 552 states and rung k=24 has 3.10e23 — 21
orders of magnitude apart, and that is the only difference.

The question was originally "where does it break", which turned out to be the
wrong question. Rung k=10 solved 0/60 at 3,000 steps and 11/60 at 40,000: it had
not broken, it was undertrained. So the measurement is a SURFACE — solve rate
against (size, budget) — from which the interesting quantity falls out: the
compute needed to reach a given competence, as a function of size. That is a
scaling law, and unlike a breakage threshold it predicts rather than reports.

Every run trains once to the largest budget and snapshots along the way, so the
extra budget points are free.

Two controls make the result mean something:

  ground truth   rungs k<=6 are enumerated exactly by exact.py, so there we
                 measure the true optimality gap, not merely a solve rate. Those
                 runs also pin the diameter, which matters because a bigger
                 space is usually a deeper one: measured, diameter = k+2 exactly,
                 so depth cannot explain a collapse spanning 21 orders of size.

  budget         see above — the reason this file measures a surface rather than
                 a line.

Resumable by construction: every checkpoint and evaluation is a file, and an
existing one is skipped. Kill it and re-run.

    ../.venv-mlx/bin/python sweep.py --steps 40000 --seeds 1
    ../.venv-mlx/bin/python sweep.py --steps 40000 --seeds 3   # adds seeds later
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from davi import rung_size

HERE = Path(__file__).parent
PY = str(Path(sys.executable))
RESULTS = HERE.parent / "eval" / "results"

RUNGS = [2, 4, 6, 8, 10, 12, 16, 24]

# The question is not "does it fail at size S" but "how much compute does size S
# need". A rung that fails at one budget may simply be undertrained — measured:
# k=10 solved 0/60 at 3k steps and 11/60 at 40k. So every run is trained once to
# the largest budget and snapshotted along the way, giving several budget points
# for the price of one run.
BUDGETS = [3_000, 12_000, 40_000]

# Held fixed across every rung. Sizing the network per rung would confound the
# measurement: a failure could then be small-net rather than large-space.
PROTOCOL = dict(batch=1024, hidden="4096,2048,1024", kmax=40,
                sync_every=100, lr=1e-3, curriculum="on", half="on")


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def train_one(rung, seed, steps, force=False):
    task, tag = f"wings-k{rung}", f"_s{seed}"
    meta = HERE / f"ckpt_{task}{tag}.json"
    if meta.exists() and not force:
        done = json.loads(meta.read_text()).get("step", 0)
        if done >= steps:
            return "cached"
    snaps = ",".join(str(b) for b in BUDGETS if b < steps)
    cmd = [PY, str(HERE / "davi.py"), "--task", task, "--tag", tag,
           "--seed", str(seed), "--steps", str(steps),
           "--snapshot-at", snaps,
           "--ckpt-every", str(max(500, steps // 10)),
           "--batch", str(PROTOCOL["batch"]), "--hidden", PROTOCOL["hidden"],
           "--kmax", str(PROTOCOL["kmax"]), "--sync-every", str(PROTOCOL["sync_every"]),
           "--lr", str(PROTOCOL["lr"]), "--curriculum", PROTOCOL["curriculum"],
           "--half", PROTOCOL["half"]]
    rc = run(cmd, HERE / f"log_train_{task}{tag}.txt")
    return "ok" if rc == 0 else f"FAILED rc={rc}"


def eval_one(rung, seed, n, width, force=False, suffix=""):
    task, tag = f"wings-k{rung}", f"_s{seed}{suffix}"
    out = RESULTS / f"eval-{task}{tag}.json"
    if out.exists() and not force:
        return "cached", json.loads(out.read_text())
    cmd = [PY, str(HERE / "evaluate.py"), "--task", task, "--tag", tag,
           "--n", str(n), "--width", str(width)]
    rc = run(cmd, HERE / f"log_eval_{task}{tag}.txt")
    if rc != 0 or not out.exists():
        return f"FAILED rc={rc}", None
    return "ok", json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rungs", default="", help="comma list; default all")
    ap.add_argument("--n", type=int, default=200, help="evaluation scrambles per run")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",")] if args.rungs else RUNGS
    plan = [(k, s) for k in rungs for s in range(args.seeds)]

    print(f"ladder sweep: {len(rungs)} rungs x {args.seeds} seeds = {len(plan)} runs")
    print(f"protocol: {args.steps:,} steps, {PROTOCOL}")
    print(f"evaluation: n={args.n} scrambles, beam width {args.width}\n")
    print(f"{'rung':>10} {'states':>10} {'seeds':>6}")
    for k in rungs:
        print(f"{'wings-k'+str(k):>10} {rung_size(k):>10.2e} {args.seeds:>6}")
    if args.dry_run:
        return

    budgets = [b for b in BUDGETS if b <= args.steps] + \
              ([args.steps] if args.steps not in BUDGETS else [])
    t0, rows = time.time(), []
    for i, (k, seed) in enumerate(plan, 1):
        head = f"[{i}/{len(plan)}] wings-k{k} s{seed}"
        st = train_one(k, seed, args.steps, args.force)
        print(f"{head}  train {st}", flush=True)
        if st.startswith("FAILED"):
            continue
        # one run, several budget points
        for b in budgets:
            suffix = "" if b == args.steps else f"@{b}"
            st, res = eval_one(k, seed, args.n, args.width, args.force, suffix)
            if res is None:
                print(f"{head}  @{b:,} eval {st}", flush=True)
                continue
            res["budget"] = b
            rows.append(res)
            opt = res.get("optimality")
            print(f"{head}  @{b:>6,}  solved {res['solve_rate']*100:5.1f}% "
                  f"[{res['ci95'][0]*100:4.1f}-{res['ci95'][1]*100:4.1f}]  "
                  + (f"optimal {opt['exact_rate']*100:5.1f}%  " if opt else "")
                  + f"slope {res['deep_slope']:+.4f}", flush=True)

    out = RESULTS / "ladder.json"
    out.write_text(json.dumps({
        "steps": args.steps, "seeds": args.seeds, "protocol": PROTOCOL,
        "budgets": BUDGETS,
        "eval_n": args.n, "eval_width": args.width,
        "elapsed_hours": (time.time() - t0) / 3600,
        "runs": rows,
    }, indent=2) + "\n")
    print(f"\n{len(rows)} runs -> {out.relative_to(HERE.parent)}  "
          f"({(time.time()-t0)/3600:.1f}h)")


if __name__ == "__main__":
    main()
