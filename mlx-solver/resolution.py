"""Where along the path to the goal does a learned heuristic stop discriminating?

This is the measurement the ladder exists to make, and the one the adjacent
literature does not. kSubS (arXiv:2108.11204) reports a signal-to-noise ratio at
a single fixed distance; AdaSubS (arXiv:2406.03361) probes degradation by
injecting synthetic Gaussian noise rather than comparing against truth. Neither
measures agreement with EXACT ground truth as a function of TRUE distance.

The operational question is not "is J accurate" but "can J still tell closer
from further". A search that only ranks candidates needs nothing more than that,
and nothing less. So the headline number here is:

    pairwise ranking accuracy at distance d
      = P[ J(s) < J(s') ]  for s at true distance d and s' at true distance d+1

At 1.0 the heuristic orders those two shells perfectly. At 0.5 it is a coin
flip and search at that depth is blind, however small the regression error is.
That distinction matters because a heuristic can have low absolute error and no
ranking power at the same time, which is exactly the failure we are chasing.

Reported alongside it: Spearman rank correlation against h* both globally and
within each shell, and the per-shell mean and spread of J, which is what shows
the dynamic range collapsing.

    ../.venv-mlx/bin/python resolution.py --task wings-k6 --tag _s0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from davi import Task
from evaluate import RESULTS, j_of, load
from exact import indexer

HERE = Path(__file__).parent


def spearman(a, b):
    """Rank correlation, ties averaged. Distances are heavily tied by
    construction (whole shells share a value) so ties must be handled or the
    coefficient is meaningless here."""
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x))
        # average ranks within tied groups
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def sample_states(task, n, walk, rng):
    """Deep random walks approximate the uniform distribution over the rung.
    Well past the diameter this is a good approximation and it needs no
    unranking machinery."""
    st = np.tile(task.solved, (n, 1))
    for _ in range(walk):
        st = task.apply(st, rng.integers(0, task.n_moves, size=n))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=40000, help="states sampled")
    ap.add_argument("--walk", type=int, default=200, help="random-walk length for sampling")
    ap.add_argument("--pairs", type=int, default=20000, help="pairs per shell boundary")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    msuf = "" if task.moveset == "all" else f"-{task.moveset}"
    if task.k is None or not (HERE / f"exact_k{task.k}{msuf}.npy").exists():
        raise SystemExit(f"no exact table for {args.task}/{task.moveset}; run exact.py --k {task.k} --moves {task.moveset} --save-table")

    exact = np.load(HERE / f"exact_k{task.k}{msuf}.npy")
    index = indexer(task.k)
    rng = np.random.default_rng(args.seed)

    states = sample_states(task, args.n, args.walk, rng)
    true_d = exact[index(states)].astype(np.int32)
    j = j_of(task, net, states)

    shells = sorted(int(d) for d in np.unique(true_d) if d > 0)
    per_shell = []
    for d in shells:
        sel = true_d == d
        per_shell.append({
            "d": d, "count": int(sel.sum()),
            "mean_j": float(j[sel].mean()), "sd_j": float(j[sel].std()),
        })

    # the headline: can J separate shell d from shell d+1?
    boundaries = []
    for d in shells[:-1]:
        a, b = j[true_d == d], j[true_d == d + 1]
        if len(a) < 50 or len(b) < 50:
            continue
        m = min(args.pairs, len(a) * 4)
        ia = rng.integers(0, len(a), size=m)
        ib = rng.integers(0, len(b), size=m)
        acc = float(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
        # the gap between shell means, in units of within-shell noise: the
        # signal-to-noise form kSubS reports, but resolved per shell
        pooled = np.sqrt((a.var() + b.var()) / 2)
        boundaries.append({
            "d": d, "acc": acc,
            "gap": float(b.mean() - a.mean()),
            "snr": float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("inf"),
        })

    res = {
        "task": args.task, "tag": args.tag, "k": task.k, "states": task.size,
        "step": meta.get("step"), "n": args.n,
        "spearman_global": spearman(j, true_d),
        "shells": per_shell, "boundaries": boundaries,
        # where does ranking fall to chance? the number the whole ladder is for
        "first_blind_d": next((b["d"] for b in boundaries if b["acc"] < 0.6), None),
    }

    print(f"{args.task}{args.tag}  k={task.k}  {task.size:.2e} states  step {res['step']:,}")
    print(f"  global Spearman(J, h*) = {res['spearman_global']:+.4f}\n")
    print(f"  {'d':>3} {'count':>7} {'mean J':>8} {'sd J':>7}   {'d->d+1 acc':>10} {'gap':>7} {'SNR':>6}")
    bmap = {b["d"]: b for b in boundaries}
    for s in per_shell:
        b = bmap.get(s["d"])
        line = f"  {s['d']:>3} {s['count']:>7,} {s['mean_j']:>8.3f} {s['sd_j']:>7.3f}"
        if b:
            flag = "  <- blind" if b["acc"] < 0.6 else ""
            line += f"   {b['acc']:>10.3f} {b['gap']:>7.3f} {b['snr']:>6.2f}{flag}"
        print(line)
    fb = res["first_blind_d"]
    print(f"\n  ranking falls below 0.6 at true distance {fb}" if fb else
          "\n  ranking stays above 0.6 at every measured distance")

    out = Path(args.out) if args.out else RESULTS / f"resolution-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
