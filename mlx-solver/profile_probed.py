"""Per-shell ranking profiles on rungs too large to enumerate.

profiles.py needs a full exact distance table, so it stops at k=6 — and every
rung it can reach solves at beam width 1, leaving nothing for a search-cost
prediction to explain. The rungs with real search cost (k=8 upward, solve rates
95% down to 0%) are exactly the ones enumeration cannot touch.

probe_exact.py closes that gap for a SAMPLE, which is all a profile needs: build
the ball of radius f around solved once, then meet each query state in the
middle. This script does that and computes the same per-shell ranking accuracy
against those exact distances.

Sampling is deliberately spread over scramble lengths rather than drawn at one
depth. States scrambled 30 moves all pile into two or three shells near the mean,
which measures the profile nowhere; sweeping the scramble length populates the
whole range so the decay curve is visible end to end. The true distance is then
measured, never assumed — a state scrambled 9 moves is usually not 9 moves from
solved, and treating scramble length as ground truth is the mistake this whole
apparatus exists to avoid.

    ../.venv-mlx/bin/python profile_probed.py --task wings-k8 --tag _s0 --forward 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from evaluate import RESULTS, j_of, load
from exact import indexer
from probe_exact import build_forward, distance

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--forward", type=int, default=5)
    ap.add_argument("--back", type=int, default=6)
    ap.add_argument("--cap", type=int, default=30_000_000)
    ap.add_argument("--per-len", type=int, default=180, help="states per scramble length")
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--min-shell", type=int, default=40, help="skip shells thinner than this")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    index = indexer(task.k)
    rng = np.random.default_rng(args.seed)

    print(f"{args.task}{args.tag}  k={task.k} ({task.moveset})  {task.size:.2e} states  "
          f"step {meta.get('step'):,}")

    t0 = time.time()
    tidx, td, f_used = build_forward(task, index, args.forward, args.cap, verbose=True)
    print(f"  ball radius {f_used}: {tidx.size:,} states ({time.time()-t0:.0f}s), "
          f"exact to distance {f_used + args.back}\n")

    # spread the sample across scramble lengths so every shell gets populated
    states, probed = [], []
    for L in range(1, args.max_len + 1):
        st = np.tile(task.solved, (args.per_len, 1))
        for _ in range(L):
            st = task.apply(st, rng.integers(0, task.n_moves, size=args.per_len))
        states.append(st)
    states = np.concatenate(states)

    t1 = time.time()
    total = states.shape[0]
    for i in range(total):
        probed.append(distance(task, index, tidx, td, states[i], args.back))
        # Report often enough that a stall is distinguishable from slow progress.
        # The first estimate lands within seconds rather than at the halfway mark.
        if (i + 1) % 20 == 0 or i + 1 == total:
            el_so_far = time.time() - t1
            rate = (i + 1) / el_so_far
            eta = (total - i - 1) / rate
            print(f"    probed {i+1:>5}/{total}  {el_so_far/(i+1):>6.1f}s each  "
                  f"ETA {eta/60:>6.1f} min", flush=True)
    el = time.time() - t1

    keep = np.array([d is not None for d in probed])
    true_d = np.array([d for d in probed if d is not None], dtype=np.int32)
    h = j_of(task, net, states[keep]).astype(np.float64)
    print(f"  probed {states.shape[0]:,} states in {el:.0f}s "
          f"({el/states.shape[0]*1000:.0f} ms each); "
          f"{(~keep).sum()} beyond reach {f_used + args.back}")

    shells, counts = np.unique(true_d, return_counts=True)
    usable = [int(d) for d, c in zip(shells, counts) if c >= args.min_shell and d > 0]
    print(f"  shells with >= {args.min_shell} states: {usable}\n")

    rows = []
    for d in usable:
        if d + 1 not in usable:
            continue
        a, b = h[true_d == d], h[true_d == d + 1]
        m = min(args.pairs, len(a) * len(b))
        ia, ib = rng.integers(0, len(a), size=m), rng.integers(0, len(b), size=m)
        acc = float(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
        rows.append({"d": d, "acc": acc, "n_lo": int(len(a)), "n_hi": int(len(b)),
                     "mean_lo": float(a.mean()), "sd_lo": float(a.std())})

    gdrc = float(kendalltau(h, true_d).statistic)
    res = {"task": args.task, "tag": args.tag, "k": task.k, "moves": task.moveset,
           "states": task.size, "step": meta.get("step"),
           "forward": f_used, "back": args.back, "reach": f_used + args.back,
           "n_probed": int(keep.sum()), "unresolved": int((~keep).sum()),
           "gdrc": gdrc, "profile": rows}

    print(f"  GDRC (pooled Kendall tau) = {gdrc:+.4f}\n")
    print(f"  {'d':>3} {'n(d)':>7} {'n(d+1)':>7} {'acc':>8}  {'mean J':>8}")
    for r in rows:
        print(f"  {r['d']:>3} {r['n_lo']:>7} {r['n_hi']:>7} {r['acc']:>8.3f}  {r['mean_lo']:>8.3f}")
    if rows:
        print(f"\n  profile {rows[0]['acc']:.3f} -> {rows[-1]['acc']:.3f} "
              f"(drop {rows[0]['acc']-rows[-1]['acc']:+.3f})")

    out = Path(args.out) if args.out else RESULTS / f"probeprofile-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
