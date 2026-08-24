"""Does the two-moment law hold outside the Rubik's cube?

This is the external-validity test the paper needs, and the honest answer is
NO, NOT CLEANLY. The law predicts adjacent-shell ordering accuracy as
Phi(gap / (sd * sqrt(2))) with nothing fitted, and on cube sub-problems it does
that to a mean absolute error of 0.012 against an estimator noise floor of
0.002. On the 8-puzzle the same measurement, through the same code path, gives
0.047. The gap is real and it is not an artifact of measurement.

WHERE IT SURVIVES. Learned heuristics: tile 0.010 against the cube's 0.007.
Pattern databases are where it breaks, at 0.062 against the cube's 0.020.

FOUR EXPLANATIONS TESTED AND REJECTED. Each is recorded because a rejected
hypothesis is the cheapest thing a later reader can be given, and three of these
are the ones anybody would reach for first.

  Saturation. A weak abstraction tops out at its own small diameter long before
  the real one, so deep shells pile up against a ceiling. Predicted: the weakest
  rungs misfit worst. Measured: exactly backwards, Spearman +1.000 between the
  abstraction-to-task diameter ratio and the error. The STRONGEST rung, k=5 at a
  ratio of 0.97, is the worst fit at 0.122; the weakest, k=1 at 0.39, is the
  best at 0.035.

  Near-determinism. A heuristic that is nearly a function of true distance has a
  within-shell spike, which is the least normal shape available. Predicted:
  error grows with d'. Measured: error FALLS with d', Spearman -0.53, reaching
  0.0003 above d' = 5.

  A strength confound. The tile's heuristics are much weaker per shell, median
  d' 0.25 against the cube's 1.24, and the law is worst at low d' in both
  domains. This is the paper's own warning about unmatched baselines pointed
  back at itself, so it had to be checked. Reweighting the cube to the tile's d'
  composition moves it from 0.0117 to 0.0114, nowhere near the tile's 0.0465,
  and within every matched d' bin below 2 the tile is 2.5x to 7x worse.

  Ties. The equal-variance binormal identity is a statement about continuous
  variables, and this project already measures a tie bias in tau-b. Measured:
  backwards again. The CUBE has a median tie rate of 0.331 and the tile 0.000,
  because tile pattern-database values spread over many integers while cube ones
  concentrate on a few.

WHAT DOES CARRY SIGNAL. Skewness of the within-shell distribution, Spearman
+0.71 against the error, and it is a partial explanation rather than a full one.
Below |skew| = 1 the two domains fit equally well, ratios 0.76, 1.18 and 1.06
across three bins. Above it the tile is still 3.15x worse, and reweighting the
cube to the tile's skew composition moves it only 0.0221 to 0.0224. Note that
equal variance, the assumption the paper actually names, is NOT the problem: the
tile's variance ratio of 1.195 is better behaved than the cube's 1.428.

So the law's parameter-free form is not domain-general. It survives on learned
heuristics in both domains and on everything at low skew, and it degrades on
skewed within-shell distributions, with a residual on the sliding tile that
these four hypotheses do not account for.

    ../.venv-mlx/bin/python cross_domain.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, skew, kurtosis, pearsonr, spearmanr

from domains import abstract, load_table, make_task, project, rank

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"
SQ2 = np.sqrt(2.0)

# (task, abstraction rungs, scramble depth). The depth is set past each task's
# diameter so the deep shells are populated rather than sampled twice.
SPECS = [("wings-k6", [2, 3, 4, 5], 20), ("tile-3x3", [1, 2, 3, 4, 5], 45)]


def shell_rows(name, ks, max_len, per_len, rng):
    """One row per (abstraction, shell boundary), with the law's inputs measured.

    Only pattern databases are used here. They are the class the domain gap
    lives in, and unlike a learned heuristic they need no checkpoint, so this
    script reproduces without a trained model on either domain.
    """
    task = make_task(name)
    exact = load_table(task)

    # Sweep the scramble length rather than walking deep. A deep walk piles
    # almost everything into the shells near the mean distance and leaves the
    # shallow half measured on a handful of states, which is the same reason
    # profiles.py sweeps.
    parts = [np.tile(task.solved, (per_len, 1))]
    cur = parts[0]
    for _ in range(max_len):
        cur = task.apply(cur, rng.integers(0, task.n_moves, size=per_len))
        parts.append(cur.copy())
    states = np.concatenate(parts)
    true_d = exact[rank(task, states)].astype(np.int32)

    rows = []
    for j in ks:
        sub = abstract(task, j)
        tab = load_table(sub)
        h = tab[rank(sub, project(task, states, j))].astype(np.float64)
        for d in range(1, int(true_d.max())):
            a, b = h[true_d == d], h[true_d == d + 1]
            if len(a) < 200 or len(b) < 200:
                continue
            m = 20000
            ia, ib = rng.integers(0, len(a), m), rng.integers(0, len(b), m)
            tie = float(np.mean(a[ia] == b[ib]))
            acc = float(np.mean(a[ia] < b[ib]) + 0.5 * tie)
            gap = float(b.mean() - a.mean())
            sd = float(np.sqrt((a.var() + b.var()) / 2))
            if sd <= 0:
                continue
            centred = np.r_[a - a.mean(), b - b.mean()]
            rows.append({
                "domain": "tile" if name.startswith("tile-") else "cube",
                "task": name, "rung": j, "d": d,
                "acc": acc, "pred": float(norm.cdf(gap / (sd * SQ2))),
                "gap": gap, "sd": sd, "dprime": gap / sd, "tie": tie,
                "skew": float(abs(skew(centred))),
                "kurt": float(abs(kurtosis(centred))),
                "var_ratio": float(max(a.var(), b.var())
                                   / max(min(a.var(), b.var()), 1e-12)),
            })
    return rows


def mae(rows):
    return float(np.mean([abs(r["acc"] - r["pred"]) for r in rows])) if rows else None


def stratify(rows, key, edges):
    """Cube against tile inside matched bins of `key`, plus the reweighted cube.

    The reweighting is the part that decides things. If the cube's error, taken
    bin by bin but weighted by how often the TILE lands in each bin, comes out
    near the tile's, then the domain gap was never a domain gap: it was the two
    sets of heuristics differing on `key`. If it does not move, the gap is real.
    """
    out = {"bins": [], "cube_mae": mae([r for r in rows if r["domain"] == "cube"]),
           "tile_mae": mae([r for r in rows if r["domain"] == "tile"])}
    num = den = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = [r for r in rows if r["domain"] == "cube" and lo <= r[key] < hi]
        t = [r for r in rows if r["domain"] == "tile" and lo <= r[key] < hi]
        cm, tm = mae(c), mae(t)
        out["bins"].append({"lo": lo, "hi": hi, "cube_n": len(c), "tile_n": len(t),
                            "cube_mae": cm, "tile_mae": tm,
                            "ratio": (tm / cm) if (cm and tm) else None})
        if cm is not None and t:
            num += cm * len(t)
            den += len(t)
    out["cube_reweighted_to_tile"] = (num / den) if den else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-len", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for name, ks, max_len in SPECS:
        rows += shell_rows(name, ks, max_len, args.per_len, rng)

    res = {"formula": "acc(d) = Phi( gap / (sd * sqrt(2)) ), no fitted parameters",
           "note": "pattern databases only; see the module docstring",
           "by_domain": {}, "correlations": {}, "stratified": {}}

    print("Two-moment law, pattern databases, both domains, one code path\n")
    for dom in ("cube", "tile"):
        rs = [r for r in rows if r["domain"] == dom]
        res["by_domain"][dom] = {
            "n": len(rs), "mae": mae(rs),
            "bias": float(np.mean([r["acc"] - r["pred"] for r in rs])),
            "median_skew": float(np.median([r["skew"] for r in rs])),
            "median_tie": float(np.median([r["tie"] for r in rs])),
            "median_dprime": float(np.median([r["dprime"] for r in rs])),
            "median_var_ratio": float(np.median([r["var_ratio"] for r in rs])),
        }
        b = res["by_domain"][dom]
        print(f"  {dom:>5}  n={b['n']:>3}  MAE {b['mae']:.4f}  bias {b['bias']:+.4f}   "
              f"median skew {b['median_skew']:.3f}  tie {b['median_tie']:.3f}  "
              f"d' {b['median_dprime']:.2f}  var ratio {b['median_var_ratio']:.3f}")

    err = np.array([abs(r["acc"] - r["pred"]) for r in rows])
    print("\n  what predicts the error, pooled over both domains:")
    for key in ("skew", "kurt", "var_ratio", "tie", "dprime"):
        x = np.array([r[key] for r in rows])
        res["correlations"][key] = {"pearson": float(pearsonr(x, err).statistic),
                                    "spearman": float(spearmanr(x, err).statistic)}
        c = res["correlations"][key]
        print(f"    |error| vs {key:>9}   Pearson {c['pearson']:+.3f}   "
              f"Spearman {c['spearman']:+.3f}")

    for key, edges in (("dprime", [0, .25, .5, 1, 2, 1e9]),
                       ("skew", [0, .4, .7, 1.0, 1e9]),
                       ("var_ratio", [1, 1.5, 3, 10, 1e9])):
        res["stratified"][key] = stratify(rows, key, edges)
        s = res["stratified"][key]
        print(f"\n  matched on {key}:")
        for b in s["bins"]:
            if not (b["cube_n"] or b["tile_n"]):
                continue
            cm = f"{b['cube_mae']:.4f}" if b["cube_mae"] is not None else "  -  "
            tm = f"{b['tile_mae']:.4f}" if b["tile_mae"] is not None else "  -  "
            rr = f"{b['ratio']:.2f}" if b["ratio"] else "  -  "
            print(f"    {b['lo']:>5g} to {b['hi']:<8g} cube n={b['cube_n']:>3} {cm}   "
                  f"tile n={b['tile_n']:>3} {tm}   ratio {rr}")
        print(f"    cube reweighted to the tile's {key} mix: "
              f"{s['cube_reweighted_to_tile']:.4f}   (tile measured {s['tile_mae']:.4f})")

    out = Path(args.out) if args.out else RESULTS / "cross-domain-law.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "rows": rows}, indent=2) + "\n")
    print(f"\n  -> {out.name}")


if __name__ == "__main__":
    main()
