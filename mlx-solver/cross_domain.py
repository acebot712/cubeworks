"""Does the two-moment law hold outside the Rubik's cube? Not everywhere, and
the boundary is not the one the domain split suggests.

The law predicts adjacent-shell ordering accuracy as Phi(gap / (sd * sqrt(2)))
with nothing fitted. Measured through one code path, on the full corpus, it does
that to a mean absolute error of 0.0117 over 252 cube observations and 0.0465
over 117 sliding-tile observations, against an estimator noise floor near 0.002.
Read as a domain split, that says the law does not generalise. Read properly, it
says something more useful.

WHAT THE ERROR ACTUALLY TRACKS. Skewness of the within-shell value distribution,
Spearman +0.74, and the relationship is monotone across all three tasks:

    task        median |skew|    MAE
    wings-k6        0.599      0.0221
    tile-3x3        0.858      0.0508
    tile-2x4        1.557      0.0718

Matched on skew, the domain gap disappears below |skew| = 1 and the tile fits
slightly BETTER than the cube in every bin: ratios 0.73, 0.70 and 0.89 across
bins 0 to 0.4, 0.4 to 0.7 and 0.7 to 1.0. Above 1.0 the tile is still 3.4 times
worse, so skew accounts for the gap over most of the range but leaves a residual
this file does not explain.

So the honest claim is not "the law is a cube fact". It is that the law is
accurate where the within-shell distribution is close to symmetric and degrades
as that distribution skews, in both state spaces. The sliding tile matters
because its pattern databases are skewed and the cube's are not, which is why
the cube alone could never have revealed the boundary.

The class breakdown says the same thing. Learned heuristics fit at 0.0103 on the
tile against 0.0068 on the cube, essentially unchanged; pattern databases fit at
0.0621 against 0.0200. Learned value functions are smooth and near-symmetric
within a shell; a coarse abstraction is not.

FOUR EXPLANATIONS TESTED AND REJECTED, three of them backwards from the
prediction. Recorded because a ruled-out explanation is the cheapest thing to
hand the next reader, and these are the ones anybody reaches for first.

  Saturation, that a weak abstraction tops out at its own small diameter long
  before the real one. Predicted the weakest rungs would misfit worst. Measured
  Spearman +1.000 the other way: the strongest rung, at a diameter ratio of
  0.97, is the worst fit at 0.122, and the weakest, at 0.39, is the best at
  0.035.

  Near-determinism, that a within-shell spike is the least normal shape
  available. Predicted error growing with d'. Measured error FALLING with d',
  Spearman -0.53, reaching 0.0003 above d' = 5.

  A strength confound. Tile heuristics are much weaker per shell, median d' 0.27
  against the cube's 0.90, and the law is worst at low d' in both domains. This
  is this project's own warning about unmatched baselines turned on itself, so
  it had to be checked. Reweighting the cube to the tile's d' composition moves
  it to 0.0160 against the tile's 0.0595, and inside matched d' bins the tile is
  still several times worse.

  Ties, which the equal-variance binormal identity does assume away, and which
  this project already measures a bias from in tau-b. Backwards again: the CUBE
  has a median tie rate of 0.325 and the tile 0.000, because tile
  pattern-database values spread over many integers while cube ones concentrate
  on a few.

Note also that equal variance, the assumption the paper names explicitly, is not
the culprit. The tile's variance ratio is 1.258 against the cube's 1.428, better
behaved, and reweighting on it closes almost none of the gap.

Pattern databases only, so this reproduces without a trained model on either
domain.

    ../.venv-mlx/bin/python cross_domain.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, skew, kurtosis, pearsonr, spearmanr

from domains import domain_of_name, load_table, make_task

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"
SQ2 = np.sqrt(2.0)

# (task, abstraction rungs, scramble depth). The depth is set past each task's
# diameter so the deep shells are populated rather than sampled twice.
SPECS = [("wings-k6", [2, 3, 4, 5], 20),
         ("tile-3x3", [1, 2, 3, 4, 5], 45),
         # A second board, to separate "the sliding tile" from "the 8-puzzle".
         # If the misfit is a property of the domain it should appear here too;
         # if it only appears on 3x3 then the finding is about one board and
         # says much less than it looks like it does.
         ("tile-2x4", [1, 2, 3, 4], 50)]


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
    true_d = exact[task.rank(states)].astype(np.int32)

    rows = []
    for j in ks:
        sub = task.abstract(j)
        tab = load_table(sub)
        h = tab[sub.rank(task.project(states, j))].astype(np.float64)
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
                "domain": domain_of_name(name),
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

    # PER BOARD. Pooling the two tile boards would let one of them carry the
    # other, and the whole point of the second board is to ask whether the
    # misfit belongs to the domain or to the 8-puzzle specifically.
    res["by_task"] = {}
    print("\n  per board, since a domain-level number can hide a single bad board:")
    for name, _, _ in SPECS:
        rs = [r for r in rows if r["task"] == name]
        if not rs:
            continue
        res["by_task"][name] = {
            "n": len(rs), "mae": mae(rs),
            "bias": float(np.mean([r["acc"] - r["pred"] for r in rs])),
            "median_skew": float(np.median([r["skew"] for r in rs])),
            "median_dprime": float(np.median([r["dprime"] for r in rs])),
        }
        b = res["by_task"][name]
        print(f"    {name:>10}  n={b['n']:>3}  MAE {b['mae']:.4f}  bias {b['bias']:+.4f}   "
              f"median skew {b['median_skew']:.3f}   d' {b['median_dprime']:.2f}")

    err = np.array([abs(r["acc"] - r["pred"]) for r in rows])
    print("\n  what predicts the error, pooled over both domains:")
    for key in ("skew", "kurt", "var_ratio", "tie", "dprime"):
        x = np.array([r[key] for r in rows])
        res["correlations"][key] = {"pearson": float(pearsonr(x, err).statistic),
                                    "spearman": float(spearmanr(x, err).statistic)}
        c = res["correlations"][key]
        print(f"    |error| vs {key:>9}   Pearson {c['pearson']:+.3f}   "
              f"Spearman {c['spearman']:+.3f}")

    # The class split comes from the full corpus rather than this script's
    # pattern-database-only rows, because the learned side needs checkpoints.
    # It is copied in here so one file carries the whole comparison.
    try:
        law = json.load(open(RESULTS / "dprime-law.json"))
        from dprime_law import load_insample
        ins = [r for r in load_insample() if "split_acc" in r]
        res["by_class"] = {}
        for cls in ("learned", "PDB", "random"):
            res["by_class"][cls] = {}
            for dom in ("cube", "tile"):
                rs = [r for r in ins if r["kind"] == cls and r["domain"] == dom]
                if rs:
                    res["by_class"][cls][dom] = {
                        "n": len(rs),
                        "mae": float(np.mean([abs(r["split_acc"] - r["split_pred"])
                                              for r in rs]))}
    except Exception as exc:                       # no profiles yet, or no checkpoints
        print(f"  (class split skipped: {exc})")

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
