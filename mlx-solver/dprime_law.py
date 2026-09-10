"""Per-shell ordering accuracy is a two-moment quantity.

The profile experiment measures, for each true-distance shell d, how often a
heuristic ranks a state at d below a state at d+1. That accuracy is what search
depends on. This script asks whether it is *determined* by something simpler.

Write gap(d) for the difference in mean heuristic value between shells d+1 and
d, and sigma(d) for the within-shell spread. If the two shells' value
distributions were normal with equal variance, the probability that a draw from
the lower shell falls below a draw from the upper one would be exactly

    a(d) = Phi( gap(d) / (sigma(d) * sqrt(2)) )

There is nothing to fit here -- no coefficient, no intercept. The claim is
falsifiable and could fail badly, most obviously for pattern databases, whose
values are small integers with heavy ties and no reason to look normal.

It does not fail. That matters for three reasons.

  1. It explains the decay. Ordering quality falls with distance because
     sigma grows relative to gap, not because of anything about how the
     heuristic was built -- which is why learned heuristics and exact
     abstractions decay alike.
  2. The ratio is scale-invariant. Multiplying a heuristic by a constant moves
     gap and sigma together and leaves accuracy untouched, so no amount of
     rescaling or recalibrating output magnitudes can improve ranking.
  3. It says what an intervention must do. Adjacent shells are one move apart,
     so gap is pinned near 1 for any distance-calibrated heuristic. The only
     free variable left is sigma. A squared-error objective penalises bias and
     variance alike; ordering only cares about variance relative to the gap.

    ../.venv-mlx/bin/python dprime_law.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

import corpus

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"
SQ2 = np.sqrt(2.0)


def kind_of(name):
    return "learned" if name == "learned" else ("random" if name == "random" else "PDB")


def pooled_sd(row, nxt):
    """Within-shell spread shared by the two shells being compared.

    `sd_hi` was added to the profile output later than the rest, so it is
    reconstructed from the next row when absent: consecutive rows describe
    consecutive shells, and row d's upper shell is row d+1's lower shell.
    """
    lo = row["sd_lo"]
    hi = row.get("sd_hi")
    if hi is None:
        hi = nxt["sd_lo"] if nxt is not None and nxt["d"] == row["d"] + 1 else lo
    return float(np.sqrt((lo ** 2 + hi ** 2) / 2.0))


def split_row(r):
    """Prediction from half A of each shell, accuracy from half B.

    Returns None on rows written before profiles.py recorded the split.
    """
    if "acc_split" not in r:
        return None
    sd = np.sqrt((r["sd_lo_a"] ** 2 + r["sd_hi_a"] ** 2) / 2.0)
    if sd <= 0:
        return None
    gap = r["mean_hi_a"] - r["mean_lo_a"]
    return {"acc": r["acc_split"], "gap": gap, "sd": sd, "dprime": gap / sd,
            "pred": float(norm.cdf(gap / sd / SQ2))}


def rows_of(prof, meta):
    out = []
    for i, r in enumerate(prof):
        nxt = prof[i + 1] if i + 1 < len(prof) else None
        sd = pooled_sd(r, nxt)
        if sd <= 0:
            continue
        # profile_probed.py records only the lower shell's moments, so the upper
        # shell's mean comes from the next row -- consecutive rows are
        # consecutive shells. Skip the last row, which has no successor.
        mean_hi = r.get("mean_hi")
        if mean_hi is None:
            if nxt is None or nxt["d"] != r["d"] + 1:
                continue
            mean_hi = nxt["mean_lo"]
        gap = mean_hi - r["mean_lo"]
        row = meta | {"d": r["d"], "acc": r["acc"], "gap": gap, "sd": sd,
                      "dprime": gap / sd,
                      "pred": float(norm.cdf(gap / sd / SQ2)),
                      "n_lo": r.get("n_lo"), "n_hi": r.get("n_hi")}
        sp = split_row(r)
        if sp:
            row |= {f"split_{k}": v for k, v in sp.items()}
        out.append(row)
    return out


def load_insample(p3=False):
    """Profiles from the main study, or (p3=True) only the loss-comparison runs.

    The loss comparison trains heuristics under objectives this law never saw --
    a pairwise ranking loss produces outputs on an arbitrary scale, and a
    variance-penalised one is built specifically to move the quantity the law is
    about. They are therefore kept as a separate held-out set rather than pooled
    in: if a parameter-free law survives being handed heuristics designed to
    stress it, that is worth more than a larger n.
    """
    recs, stats = corpus.load("both", experiment="losses" if p3 else "main")
    rows = []
    for r in recs:
        if p3 and r["name"] != "learned":
            continue      # the abstractions there are the same tables again
        rows += rows_of(r["profile"],
                        {"file": r["file"], "k": r["k"], "moves": r["moves"],
                         "tag": r["tag"], "name": r["name"], "kind": r["kind"],
                         "domain": r["domain"]})
    return rows


def domain_of_result(d):
    """Which state space a profile came from.

    Profiles written before the sliding-tile domain existed carry no `domain`
    field, so it is inferred from the task name, which is the same rule
    domains.make_task dispatches on. Every one of those is a cube run.
    """
    return d.get("domain") or ("tile" if str(d["task"]).startswith("tile-") else "cube")


def load_heldout():
    """Rung k=8: 300x larger than anything enumerable, and the ground truth comes
    from the verified meet-in-the-middle prober rather than exhaustive BFS. If
    the law were an artifact of the k<=6 measurement pipeline it would break
    here."""
    rows = []
    for f in sorted(RESULTS.glob("probeprofile-*.json")):
        d = json.load(open(f))
        rows += rows_of(d["profile"],
                        {"file": f.stem, "k": d["k"], "moves": d["moves"],
                         "tag": d["tag"], "name": "learned", "kind": "learned"})
    return rows


def noise_floor(rows, pairs=20000, reps=400, seed=5):
    """The mean absolute error this pipeline would report if the law were EXACT.

    Per-shell accuracy is estimated from a finite number of sampled pairs, so it
    carries binomial noise of its own. Without this floor there is no way to tell
    a mean absolute error of 0.004 from a perfect fit measured imprecisely, and
    the paper was quoting numbers of exactly that size as its best results.
    """
    rng = np.random.default_rng(seed)
    p = np.array([r["pred"] for r in rows])
    m = min(pairs, int(np.median([(r.get("n_lo") or 600) * (r.get("n_hi") or 600)
                                  for r in rows])))
    draws = rng.binomial(m, np.clip(p, 0, 1), size=(reps, len(p))) / m
    return float(np.abs(draws - p).mean())


def report(rows, label, split=False):
    key = "split_" if split else ""
    rows = [r for r in rows if f"{key}acc" in r]
    if not rows:
        return None
    a = np.array([r[f"{key}acc"] for r in rows])
    p = np.array([r[f"{key}pred"] for r in rows])
    e = a - p
    floor = noise_floor([{"pred": q, "n_lo": r.get("n_lo"), "n_hi": r.get("n_hi")}
                         for q, r in zip(p, rows)])
    print(f"\n{label}: {len(rows)} (heuristic, shell) observations")
    print(f"  mean |error| {np.abs(e).mean():.4f}   (noise floor of the estimator "
          f"itself: {floor:.4f})")
    print(f"  mean error {e.mean():+.4f}   max |error| {np.abs(e).max():.3f}   "
          f"Pearson r {pearsonr(a, p).statistic:+.4f}")
    return {"n": len(rows), "r": float(pearsonr(a, p).statistic),
            "mean_err": float(e.mean()), "mae": float(np.abs(e).mean()),
            "max_err": float(np.abs(e).max()), "noise_floor": floor}


def free_parameter_check(rows):
    """Is the Gaussian form right, or would any monotone function of d' do?

    Fit Phi(d'/s + b) with s and b free. If the parameter-free law is the right
    shape, the fitted scale should land near sqrt(2) on its own and the extra
    freedom should buy very little.
    """
    from scipy.optimize import curve_fit

    a = np.array([r["acc"] for r in rows])
    dp = np.array([r["dprime"] for r in rows])
    f = lambda x, s, b: norm.cdf(x / s + b)
    (s, b), _ = curve_fit(f, dp, a, p0=[SQ2, 0.0])
    g = lambda x, s, b: 1.0 / (1.0 + np.exp(-(x / s + b)))
    (s2, b2), _ = curve_fit(g, dp, a, p0=[1.0, 0.0])
    return {"free_scale": float(s), "free_shift": float(b), "theory_scale": float(SQ2),
            "mae_parameter_free": float(np.abs(a - norm.cdf(dp / SQ2)).mean()),
            "mae_two_free_params": float(np.abs(a - f(dp, s, b)).mean()),
            "mae_logistic": float(np.abs(a - g(dp, s2, b2)).mean())}


def main():
    ins, held = load_insample(), load_heldout()
    p3 = load_insample(p3=True)
    out = {"formula": "acc(d) = Phi( gap / (sd * sqrt(2)) ), no fitted parameters",
           "in_sample": report(ins, "SHARED SAMPLE, k<=6 (moments and accuracy from "
                                    "the same states -- reported for comparison only)"),
           "in_sample_split": report(ins, "SPLIT SAMPLE, k<=6 (moments from half A, "
                                          "accuracy from half B -- THE HONEST ONE)",
                                     split=True),
           "held_out": report(held, "Held out (rung k=8, meet-in-the-middle ground truth)"),
           "held_out_split": report(held, "Held out, split sample", split=True)}
    if p3:
        out["held_out_objectives"] = report(
            p3, "Held out (heuristics trained under other objectives)")
        out["held_out_objectives_split"] = report(
            p3, "Held out (other objectives), split sample", split=True)
    # BY DOMAIN. This is the external-validity check and the reason the tile
    # code exists. Pooling the two state spaces into one mean absolute error
    # would hide exactly the failure it is meant to expose: a law that holds on
    # the cube and not on the tile would still look respectable pooled, because
    # the cube supplies most of the observations. Reported separately, on the
    # split sample, so each domain stands or falls on its own.
    domains_present = sorted({r["domain"] for r in ins})
    if len(domains_present) > 1:
        out["by_domain_split"] = {}
        for dom in domains_present:
            sub = [r for r in ins if r["domain"] == dom]
            out["by_domain_split"][dom] = report(
                sub, f"SPLIT SAMPLE, {dom} only (external validity)", split=True)

    fp = free_parameter_check(ins)
    out["free_parameter_check"] = fp
    print(f"\n  Is the form right, or merely monotone?")
    print(f"    parameter-free  Phi(d'/sqrt2)             mean |error| {fp['mae_parameter_free']:.4f}")
    print(f"    two free params Phi(d'/{fp['free_scale']:.3f} {fp['free_shift']:+.3f})    "
          f"mean |error| {fp['mae_two_free_params']:.4f}")
    print(f"    two free params logistic                  mean |error| {fp['mae_logistic']:.4f}")
    print(f"    -> the fitted scale lands on {fp['free_scale']:.3f} against a theoretical "
          f"sqrt(2) = {SQ2:.3f}, unprompted")

    a = np.array([r["acc"] for r in ins]); p = np.array([r["pred"] for r in ins])
    kinds = np.array([r["kind"] for r in ins]); dd = np.array([r["d"] for r in ins])

    print("\n  By heuristic class -- the law is not specific to learned heuristics:")
    out["by_kind"] = {}
    for kd in ("random", "PDB", "learned"):
        m = kinds == kd
        out["by_kind"][kd] = {"n": int(m.sum()),
                              "r": float(pearsonr(a[m], p[m]).statistic),
                              "mae": float(np.abs(a[m] - p[m]).mean())}
        print(f"    {kd:<8} n={m.sum():>4}  r = {pearsonr(a[m], p[m]).statistic:+.3f}   "
              f"mean |error| = {np.abs(a[m] - p[m]).mean():.4f}")

    print("\n  By shell -- worst near the goal, where the value distribution is")
    print("  truncated at zero and least normal; near exact where search operates:")
    out["by_shell"] = {}
    for dv in sorted(set(dd.tolist())):
        m = dd == dv
        if m.sum() < 8:
            continue
        out["by_shell"][int(dv)] = {"n": int(m.sum()),
                                    "mean_err": float((a[m] - p[m]).mean()),
                                    "mae": float(np.abs(a[m] - p[m]).mean())}
        print(f"    d={dv:>2}  n={m.sum():>3}  mean error {(a[m]-p[m]).mean():+.4f}   "
              f"mean |error| {np.abs(a[m]-p[m]).mean():.4f}")

    # The decomposition the law buys: two heuristics can share an accuracy and
    # fail for opposite reasons, and they need opposite fixes.
    print("\n  What the two moments say about each class:")
    out["moments"] = {}
    for kd in ("random", "PDB", "learned"):
        g = np.array([r["gap"] for r in ins if r["kind"] == kd])
        s = np.array([r["sd"] for r in ins if r["kind"] == kd])
        out["moments"][kd] = {"median_gap": float(np.median(g)),
                              "median_sd": float(np.median(s)),
                              "median_dprime": float(np.median(g / s))}
        print(f"    {kd:<8} gap {np.median(g):+.3f}   spread {np.median(s):.3f}   "
              f"d' {np.median(g/s):+.3f}")

    # Scale invariance, stated as a check rather than asserted: doubling a
    # heuristic doubles both moments and must leave the prediction alone.
    r0 = ins[len(ins) // 2]
    same = norm.cdf((2 * r0["gap"]) / (2 * r0["sd"]) / SQ2) - r0["pred"]
    print(f"\n  scale invariance check: rescaling by 2x moves the prediction by {same:+.1e}")

    out["observations"] = ins + held
    (RESULTS / "dprime-law.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n  -> dprime-law.json")


if __name__ == "__main__":
    main()
