"""How much search does a given heuristic need?

Everything the field reports (solve rate, solution length, nodes expanded) is a
property of (heuristic x search algorithm x budget), not of the heuristic. Change
the beam width and every one of those numbers moves. So none of them can answer
"is this heuristic good", and none can be compared across papers that chose
different budgets.

This script inverts the question. Instead of fixing the budget and reporting the
solve rate, it sweeps the budget and reports the SMALLEST width that reaches a
target solve rate. That number is a property of the heuristic and the problem,
with the search budget divided out.

Paired with resolution.py, which measures ranking quality intrinsically, against
exact ground truth, stratified by true distance, it gives the two halves of the
question we actually care about:

    does an intrinsic property of the heuristic predict the search it will need?

If it does, you can measure a trained heuristic cheaply and forecast its search
cost without ever running the search. Chrestien et al. (NeurIPS 2023) proved
ranking is the property that governs efficiency; this measures whether that
translates into a usable quantitative prediction.

    ../.venv-mlx/bin/python budget_curve.py --task wings-k4 --tag _dv-oi-q4_s0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluate import RESULTS, beam_solve, load, wilson

HERE = Path(__file__).parent
WIDTHS = [1, 2, 5, 10, 25, 50, 100, 250, 600, 1500]


def required_width(rows, target):
    """Smallest width reaching `target` solve rate, log-interpolated between the
    bracketing measurements. Returns None if even the widest never gets there, which is itself the answer, and must not be silently reported as the cap."""
    below = None
    for r in rows:
        if r["rate"] >= target:
            if below is None:
                return float(r["width"])
            # interpolate in log-width, where the curve is closer to linear
            x0, y0 = np.log(below["width"]), below["rate"]
            x1, y1 = np.log(r["width"]), r["rate"]
            if y1 == y0:
                return float(r["width"])
            return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
        below = r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--scramble", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=60)
    ap.add_argument("--widths", default="")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    kmax = args.scramble or meta.get("kmax", 40)
    widths = [int(w) for w in args.widths.split(",")] if args.widths else WIDTHS

    # One fixed set of states for every width, so the curve reflects the width
    # and not a fresh sample each time.
    rng = np.random.default_rng(args.seed)
    states = np.tile(task.solved, (args.n, 1))
    for _ in range(kmax):
        states = task.apply(states, rng.integers(0, task.n_moves, size=args.n))

    print(f"{args.task}{args.tag}  moves={task.moveset}  {task.size:.2e} states  "
          f"step {meta.get('step'):,}")
    print(f"  {args.n} states scrambled {kmax} moves, identical across widths\n")
    print(f"  {'width':>7} {'solved':>8} {'95% CI':>16} {'mean len':>9} {'s/state':>8}")

    rows = []
    for w in widths:
        t0 = time.time()
        solved, lens = 0, []
        for i in range(args.n):
            sol = beam_solve(task, net, states[i], w, args.max_depth)
            if sol is None:
                continue
            solved += 1
            lens.append(len(sol))
        p, lo, hi = wilson(solved, args.n)
        el = (time.time() - t0) / args.n
        rows.append({"width": w, "solved": solved, "rate": p, "ci95": [lo, hi],
                     "mean_len": float(np.mean(lens)) if lens else None,
                     "sec_per_state": el})
        print(f"  {w:>7} {solved:>4}/{args.n:<3} {f'{lo*100:.0f}-{hi*100:.0f}%':>16} "
              f"{(f'{np.mean(lens):.1f}' if lens else '-'):>9} {el:>8.2f}")
        # nothing to learn from wider search once it is solving everything
        if p >= 0.995:
            break

    res = {
        "task": args.task, "tag": args.tag, "moves": task.moveset,
        "k": task.k, "states": task.size, "step": meta.get("step"),
        "kmax": kmax, "n": args.n, "curve": rows,
        "w50": required_width(rows, 0.50),
        "w90": required_width(rows, 0.90),
    }
    print(f"\n  width for 50% solve: {res['w50'] if res['w50'] else 'never reached'}")
    print(f"  width for 90% solve: {res['w90'] if res['w90'] else 'never reached'}")

    out = Path(args.out) if args.out else RESULTS / f"budget-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
