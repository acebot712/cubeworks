"""Does a learned heuristic degrade with distance in a way a classical one does not?

This is the paper's mechanism experiment, and it is a direct comparison against
the closest prior measure.

Wilt & Ruml (JAIR 57, 2016) introduced Goal Distance Rank Correlation: Kendall's
tau between a heuristic and exact distance-to-goal, computed offline, shown to
predict search cost. It is a single pooled scalar over all states. That is the
right object for the heuristics they studied: pattern databases, whose quality
has no particular reason to vary with distance from the goal.

A heuristic trained by bootstrapping is different by construction. Value is
anchored only at the goal and propagates outward, so accuracy should decay with
distance. A pooled tau averages that decay away: two heuristics with the same
GDRC can have completely different profiles, one uniform and one collapsing
exactly where search operates.

So this script computes, on the same rung and against the same exact ground truth:

  learned    the DAVI-trained network
  PDB        a genuine pattern database, the exact distance table of a SMALLER
             rung, which is admissible on the larger one because tracking fewer
             pieces can only shorten the required solution
  random     a control, to show what a flat-at-chance profile looks like

and reports each one's pooled GDRC alongside its per-shell ranking profile. The
claim is falsified if the PDB's profile decays like the learned one's, or if the
learned profile is flat.

    ../.venv-mlx/bin/python profiles.py --task wings-k6 --tag _s0 --pdb-k 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from davi import DONT_CARE, Task
from evaluate import RESULTS, j_of, load
from exact import indexer
from resolution import sample_states

HERE = Path(__file__).parent


def project(states, j):
    """A rung-k state seen as a rung-j state: keep pieces 0..j-1, forget the rest.

    This is exactly a pattern-database abstraction, so the resulting distance is
    an admissible lower bound on the true rung-k distance.
    """
    out = states.copy()
    out[out >= j] = DONT_CARE
    return out


def sweep_sample(task, per_len, max_len, rng):
    """States drawn across a range of scramble lengths, not from deep walks.

    Deep random walks approximate the uniform distribution over the rung, which
    sounds right but concentrates almost every state into the two or three
    shells nearest the mean distance, so the shallow half of the profile is
    measured on a handful of states or not at all. Sweeping the scramble length
    populates the whole range.

    The scramble length is NOT the label. True distance is looked up in the
    exact table afterwards: a state scrambled 9 moves is usually nearer than 9,
    and treating the two as the same is the error this measurement exists to
    avoid. This matches profile_probed.py so profiles from the two are
    comparable.
    """
    out = []
    for L in range(1, max_len + 1):
        st = np.tile(task.solved, (per_len, 1))
        for _ in range(L):
            st = task.apply(st, rng.integers(0, task.n_moves, size=per_len))
        out.append(st)
    return np.concatenate(out)


def profile(h, true_d, shells, rng, pairs=20000):
    """Per-shell ranking accuracy: P[h(s) < h(s')] for s at d and s' at d+1.

    Alongside the accuracy we record the two moments of each shell. They are not
    decoration: `acc` turns out to be predicted, with no free parameters, by
    Phi(gap / (sd * sqrt(2))) -- the probability that one draw beats another
    under equal-variance normals. See dprime_law.py. So the pair (between-shell
    gap, within-shell spread) is the compressed form of the whole profile, and
    it is what an intervention has to move.
    """
    rows = []
    for d in shells[:-1]:
        sel_lo, sel_hi = true_d == d, true_d == d + 1
        a, b = h[sel_lo], h[sel_hi]
        if len(a) < 50 or len(b) < 50:
            continue
        m = min(pairs, len(a) * len(b))
        ia, ib = rng.integers(0, len(a), size=m), rng.integers(0, len(b), size=m)
        acc = float(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
        row = {"d": d, "acc": acc,
               "mean_lo": float(a.mean()), "mean_hi": float(b.mean()),
               "sd_lo": float(a.std()), "sd_hi": float(b.std()),
               "n_lo": int(a.size), "n_hi": int(b.size)}

        # SPLIT-SAMPLE VERSION. `acc` above and the moments beside it are computed
        # from the same states, and that shared sample manufactures agreement: run
        # the estimator on a heuristic with no signal at all and it reports a
        # correlation near +0.95 where the truth is zero, because a shell whose
        # sampled mean happens to sit high produces both a high predicted accuracy
        # and a high measured one. Splitting each shell in half -- moments from A,
        # accuracy from B -- removes that path entirely. The split figures are the
        # ones any claim about the two-moment law should be based on.
        half = np.zeros(true_d.shape[0], dtype=bool)
        half[::2] = True
        aA, bA = h[sel_lo & half], h[sel_hi & half]
        aB, bB = h[sel_lo & ~half], h[sel_hi & ~half]
        if min(aA.size, bA.size, aB.size, bB.size) >= 25 and aA.std() + bA.std() > 0:
            mm = min(pairs, aB.size * bB.size)
            ja, jb = rng.integers(0, aB.size, size=mm), rng.integers(0, bB.size, size=mm)
            row |= {"acc_split": float(np.mean(aB[ja] < bB[jb])
                                       + 0.5 * np.mean(aB[ja] == bB[jb])),
                    "mean_lo_a": float(aA.mean()), "mean_hi_a": float(bA.mean()),
                    "sd_lo_a": float(aA.std()), "sd_hi_a": float(bA.std())}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pdb-k", default="", help="comma list of abstraction rungs; "
                    "default = every available one. Reporting all of them is the point: "
                    "a single weak abstraction looks flat merely because it is near chance, "
                    "so the comparison must be made at matched overall strength.")
    ap.add_argument("--sampler", choices=["sweep", "walk"], default="sweep",
                    help="sweep = states across scramble lengths 1..max-len, which "
                         "populates every shell and matches profile_probed.py; "
                         "walk = deep random walks, which concentrate near the mean")
    ap.add_argument("--per-len", type=int, default=2500, help="states per scramble length")
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--n", type=int, default=40000, help="walk sampler only")
    ap.add_argument("--walk", type=int, default=200, help="walk sampler only")
    ap.add_argument("--min-shell", type=int, default=50, help="skip shells thinner than this")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    msuf = "" if task.moveset == "all" else f"-{task.moveset}"
    tbl = HERE / f"exact_k{task.k}{msuf}.npy"
    if not tbl.exists():
        raise SystemExit(f"need {tbl.name}; run exact.py --k {task.k} --moves {task.moveset} --save-table")

    if args.pdb_k:
        pdb_ks = [int(x) for x in args.pdb_k.split(",")]
    else:
        pdb_ks = [j for j in range(2, task.k)
                  if (HERE / f"exact_k{j}{msuf}.npy").exists()]
    if not pdb_ks:
        raise SystemExit("no abstraction tables available for a PDB baseline")

    rng = np.random.default_rng(args.seed)
    exact, index = np.load(tbl), indexer(task.k)

    states = (sweep_sample(task, args.per_len, args.max_len, rng)
              if args.sampler == "sweep"
              else sample_states(task, args.n, args.walk, rng))
    true_d = exact[index(states)].astype(np.int32)

    hs = {"learned": j_of(task, net, states).astype(np.float64)}
    for j in pdb_ks:
        tab = np.load(HERE / f"exact_k{j}{msuf}.npy")
        hs[f"PDB(k={j})"] = tab[indexer(j)(project(states, j))].astype(np.float64)
    hs["random"] = rng.random(states.shape[0])

    uniq, cnt = np.unique(true_d, return_counts=True)
    shells = [int(d) for d, c in zip(uniq, cnt) if d > 0 and c >= args.min_shell]
    res = {"task": args.task, "tag": args.tag, "k": task.k, "moves": task.moveset,
           "states": task.size, "step": meta.get("step"), "pdb_k": pdb_ks,
           "sampler": args.sampler, "n": int(states.shape[0]),
           "shells": shells, "heuristics": {}}

    print(f"{args.task}{args.tag}  k={task.k} ({task.moveset})  {task.size:.2e} states  "
          f"step {meta.get('step'):,}")
    print(f"  {states.shape[0]:,} states ({args.sampler} sampler), exact ground truth, "
          f"PDB abstractions: {pdb_ks}")
    print(f"  shells with >= {args.min_shell} states: {shells}\n")

    for name, h in hs.items():
        # GDRC as Wilt & Ruml define it: one pooled Kendall tau over everything
        tau = float(kendalltau(h, true_d).statistic)
        rows = profile(h, true_d, shells, rng)
        # Span is recorded as context, NOT as a health check: a heuristic with a
        # small span and clean ordering is fine. What breaks a heuristic is
        # within-shell spread exceeding the between-shell gap, and the only thing
        # that measures is the adjacent-shell accuracy in `rows` below.
        span = float(np.ptp(h))
        res["heuristics"][name] = {"gdrc": tau, "span": span, "profile": rows}
        accs = [r["acc"] for r in rows]
        drop = (accs[0] - accs[-1]) if len(accs) > 1 else 0.0
        print(f"  {name:<12} GDRC {tau:+.4f}  span {span:8.3f}  "
              f"per-shell acc {accs[0]:.3f} -> {accs[-1]:.3f}  (drop {drop:+.3f})"
              + (f"   mean {np.mean(accs):.3f}"))

    print(f"\n  {'d':>3}  " + "".join(f"{n:>14}" for n in hs))
    for i, d in enumerate(shells[:-1]):
        line = f"  {d:>3}  "
        for n in hs:
            rows = res["heuristics"][n]["profile"]
            m = next((r for r in rows if r["d"] == d), None)
            line += f"{m['acc']:>14.3f}" if m else f"{'-':>14}"
        print(line)

    out = Path(args.out) if args.out else RESULTS / f"profile-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  -> {out.name}")


if __name__ == "__main__":
    main()
