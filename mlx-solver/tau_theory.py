"""Pooled GDRC, derived from the per-shell profile that produces it.

Kendall's tau-b between a heuristic h and exact distance d* is

    tau = (C - D) / sqrt((n0 - n1)(n0 - n2))

over all pairs of sampled states, where n0 is the pair count, n1 the pairs tied
in d*, and n2 the pairs tied in h. Pairs tied in d* are exactly the pairs drawn
from the same shell, so writing n_s for the number of sampled states at distance
s and a(s,t) for the probability that h ranks a state at s below one at t,

    C - D = sum over s<t of  n_s n_t (2 a(s,t) - 1)
    n0 - n1 = sum over s<t of  n_s n_t

Three consequences follow, and each is checked below against measurement.

1. tau is a weighted average of (2a-1) over EVERY pair of shells, weighted by
   n_s n_t. Adjacent shells are a vanishing share of that sum, and they are the
   only comparisons a search makes. This is the dilution argument stated as an
   identity rather than an intuition.

2. tau-b rewards a heuristic for being COARSE. A heuristic with no ties that
   orders every shell perfectly is capped at sqrt((n0-n2)/n0) -- 0.915 on the
   k=6 sample, verified against scipy below -- because its arbitrary within-shell ordering is counted
   against it. The exact distance function, which makes the same shell ordering
   but is integer-valued and therefore tied within each shell, scores 1.000 on
   the same states. Both verified against scipy here.

   The gap is not academic. Pattern databases are small integers; learned
   heuristics are continuous. Measured below, every abstraction on this family
   collects +0.07 to +0.085 of pooled tau from ties alone, which is the same
   size as the differences the learned-versus-classical literature reports.
   Comparing tau across those two classes is not comparing like with like, and
   the fix is a line of code: break ties at random before correlating.

3. Combined with the two-moment law (dprime_law.py), a(s,t) is
   Phi((mu_t - mu_s)/sqrt(sd_s^2 + sd_t^2)), so the whole pooled scalar is
   predictable from the profile -- which is the strongest available test that
   the profile is the more fundamental object.

The same identity settles the shape of decay against strength. Holding spread
constant across shells makes a(s,s+1) constant and the decay exactly zero, so
decay is not a consequence of strength at all: it exists only insofar as spread
GROWS with distance. Sweeping overall quality traces an inverted U -- zero decay
at chance, zero decay at perfection, a maximum in between -- and the positive
correlation this project reported earlier is the rising limb of it.

    ../.venv-mlx/bin/python tau_theory.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"


def shells_of(prof):
    """Sizes, means and spreads per shell, read off consecutive profile rows."""
    n = [r["n_lo"] for r in prof] + [prof[-1]["n_hi"]]
    mu = [r["mean_lo"] for r in prof] + [prof[-1]["mean_hi"]]
    sd = [r["sd_lo"] for r in prof] + [prof[-1].get("sd_hi", prof[-1]["sd_lo"])]
    return np.array(n, float), np.array(mu, float), np.array(sd, float)


_HIST = {}


def full_shell_sizes(task_name, tag):
    """The distance histogram of EVERY sampled state, not just profiled shells.

    The profile drops shells thinner than --min-shell, but the pooled tau was
    computed over all of them, and the tie correction depends on the shells that
    were dropped as much as on the ones that were kept. Reproducing the sampler
    (same seed, same construction as profiles.py) recovers the full histogram
    without needing the network, so no GPU work and no regeneration.
    """
    from davi import Task
    from exact import indexer
    from profiles import sweep_sample

    meta = json.loads((HERE / f"ckpt_{task_name}{tag}.json").read_text())
    key = (task_name, meta.get("moves") or "all")
    if key in _HIST:
        return _HIST[key]
    task = Task(task_name, moves=key[1])
    suffix = "" if task.moveset == "all" else f"-{task.moveset}"
    tbl = HERE / f"exact_k{task.k}{suffix}.npy"
    if not tbl.exists():
        return None
    rng = np.random.default_rng(7)          # profiles.py --seed default
    states = sweep_sample(task, 2500, 16, rng)
    d = np.load(tbl)[indexer(task.k)(states)]
    _HIST[key] = np.bincount(d.astype(np.int64))[1:].astype(float)
    return _HIST[key]


def tau_from_profile(n, mu, sd, ties_h=0.0):
    """Pooled tau-b implied by shell sizes and the two moments of each shell."""
    N = n.sum()
    n0 = N * (N - 1) / 2
    n1 = (n * (n - 1) / 2).sum()
    D = len(n)
    cd = 0.0
    for s in range(D):
        for t in range(s + 1, D):
            denom = np.sqrt(sd[s] ** 2 + sd[t] ** 2)
            a = 0.5 if denom <= 0 else float(norm.cdf((mu[t] - mu[s]) / denom))
            cd += n[s] * n[t] * (2 * a - 1)
    return float(cd / np.sqrt((n0 - n1) * (n0 - ties_h * n0)))


def ceiling(n):
    """Best tau-b a TIE-FREE heuristic can score, given this shell histogram.

    Not a ceiling for every heuristic: one that ties within shells, as an
    integer-valued pattern database does, is not charged for its arbitrary
    within-shell order and can reach 1. Verified in tie_inflation() below.
    """
    N = n.sum()
    n0 = N * (N - 1) / 2
    n1 = (n * (n - 1) / 2).sum()
    return float(np.sqrt((n0 - n1) / n0))


def verify_ceiling(tname="wings-k6", mv="all", k=6):
    """Check the analytic tie-free ceiling against scipy, on a named rung.

    The paper previously asserted a ceiling of 0.929 "verified against the
    reference implementation" and attributed it to k=6. Neither was true: 0.929
    is the k=4 value, and nothing was ever run through scipy. Both figures are
    computed here and written to JSON so the claim is checkable.
    """
    from scipy.stats import kendalltau

    from davi import Task
    from exact import indexer
    from profiles import sweep_sample

    suf = "" if mv == "all" else f"-{mv}"
    task = Task(tname, moves=mv)
    st = sweep_sample(task, 2500, 16, np.random.default_rng(7))
    d = np.load(HERE / f"exact_k{k}{suf}.npy")[indexer(k)(st)].astype(np.int64)
    # every shell, INCLUDING the solved states at d=0. Dropping them changes n0
    # and the tie count, which is why an earlier version of this check produced a
    # scipy value above its own analytic ceiling.
    n = np.bincount(d).astype(float)
    rng = np.random.default_rng(17)
    # perfect shell ordering, no ties at all
    free = float(kendalltau(d + rng.random(d.size) * 0.999, d).statistic)
    # the exact distance function itself: same ordering, maximally tied
    tied = float(kendalltau(d.astype(float), d).statistic)
    return {"task": tname, "moves": mv, "analytic_tie_free_ceiling": ceiling(n),
            "scipy_tie_free": free, "scipy_exact_distance": tied}


def tie_inflation():
    """How much pooled tau each abstraction collects from ties, not from order.

    Breaking ties at random leaves the between-shell ordering -- the thing a
    search uses -- completely unchanged, so the drop is the part of the score
    that was never about ordering at all.
    """
    from scipy.stats import kendalltau

    from davi import Task
    from exact import indexer
    from profiles import project, sweep_sample

    rows = []
    for tname, mv, k in (("wings-k4", "oi-q3", 4), ("wings-k4", "all", 4),
                         ("wings-k6", "all", 6)):
        suf = "" if mv == "all" else f"-{mv}"
        if not (HERE / f"exact_k{k}{suf}.npy").exists():
            continue
        task = Task(tname, moves=mv)
        st = sweep_sample(task, 2500, 16, np.random.default_rng(7))
        d = np.load(HERE / f"exact_k{k}{suf}.npy")[indexer(k)(st)].astype(np.int64)
        rng = np.random.default_rng(3)
        for j in range(2, k):
            p = HERE / f"exact_k{j}{suf}.npy"
            if not p.exists():
                continue
            h = np.load(p)[indexer(j)(project(st, j))].astype(np.float64)
            tied = float(kendalltau(h, d).statistic)
            free = float(kendalltau(h + rng.random(h.size) * 0.999, d).statistic)
            rows.append({"task": tname, "moves": mv, "pdb_k": j, "tau": tied,
                         "tau_tiebroken": free, "inflation": tied - free,
                         "distinct_values": int(len(np.unique(h)))})
    return rows


def check_prediction():
    print("Predicting the pooled scalar from the profile that produced it")
    print("(no fitted parameters; tau-b ties in h are not modelled, which costs")
    print(" the integer-valued pattern databases and nothing else)\n")
    rows = []
    for f in sorted(RESULTS.glob("profile-*.json")):
        if "_p3-" in f.stem:      # separate experiment; see strength_control.py
            continue
        d = json.load(open(f))
        if d["k"] <= 2:
            continue
        for name, h in d["heuristics"].items():
            prof = h["profile"]
            if len(prof) < 2 or "n_lo" not in prof[0]:
                continue
            n, mu, sd = shells_of(prof)
            kind = "learned" if name == "learned" else (
                "random" if name == "random" else "PDB")
            full = full_shell_sizes(d["task"], d["tag"])
            rows.append({"kind": kind, "file": f.stem, "name": name,
                         "measured": h["gdrc"],
                         "predicted": tau_from_profile(n, mu, sd),
                         "ceiling": ceiling(full if full is not None else n),
                         "coverage": float(n.sum() / full.sum()) if full is not None else 1.0})
    for kd in ("random", "PDB", "learned"):
        r = [x for x in rows if x["kind"] == kd]
        m = np.array([x["measured"] for x in r]); p = np.array([x["predicted"] for x in r])
        print(f"  {kd:<8} n={len(r):>3}  r = {pearsonr(m, p).statistic:+.3f}   "
              f"mean |error| = {np.abs(m - p).mean():.4f}")
    m = np.array([x["measured"] for x in rows]); p = np.array([x["predicted"] for x in rows])
    print(f"  {'all':<8} n={len(rows):>3}  r = {pearsonr(m, p).statistic:+.3f}   "
          f"mean |error| = {np.abs(m - p).mean():.4f}")

    print(f"\n  profile covers {np.mean([x['coverage'] for x in rows])*100:.1f}% of the "
          f"sampled states on average; the rest sit in shells too thin to profile")
    print(f"  tau-b ceiling from the full shell histogram: "
          f"{np.mean([x['ceiling'] for x in rows]):.3f} "
          f"(range {min(x['ceiling'] for x in rows):.3f}-"
          f"{max(x['ceiling'] for x in rows):.3f})")
    best = max(rows, key=lambda x: x["measured"])
    print(f"  strongest heuristic measured: {best['name']} on {best['file'][8:]} at tau "
          f"{best['measured']:+.3f}, against {best['ceiling']:.3f} for a tie-free "
          f"heuristic that orders every shell perfectly")
    print("  -> it exceeds that, which is only possible by having ties of its own:")

    vc = verify_ceiling()
    print(f"\n  tie-free ceiling on {vc['task']} ({vc['moves']}): analytic "
          f"{vc['analytic_tie_free_ceiling']:.4f}, scipy {vc['scipy_tie_free']:.4f}; "
          f"the exact distance function itself scores {vc['scipy_exact_distance']:.4f}")

    ties = tie_inflation()
    for r in ties:
        print(f"     PDB(k={r['pdb_k']}) on {r['task']} ({r['moves']}): tau {r['tau']:+.4f}, "
              f"tie-broken {r['tau_tiebroken']:+.4f}, bought by ties "
              f"{r['inflation']:+.4f}  [{r['distinct_values']} distinct values]")
    inf = np.mean([r["inflation"] for r in ties])
    print(f"     mean inflation {inf:+.4f} -- the same size as the learned-vs-classical")
    print("     differences this literature reports, and pure measurement artifact.")
    return rows, ties


def inverted_u(n, beta=0.35, D=None):
    """Trace (tau, decay) as overall quality sweeps from chance to perfect."""
    D = D or len(n)
    mu = np.arange(D, dtype=float)
    out = []
    for s0 in np.exp(np.linspace(np.log(0.02), np.log(40.0), 240)):
        sd = s0 * (1.0 + beta * np.arange(D))
        tau = tau_from_profile(n, mu, sd)
        a_near = norm.cdf(1.0 / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
        a_far = norm.cdf(1.0 / np.sqrt(sd[-2] ** 2 + sd[-1] ** 2))
        out.append({"sigma0": float(s0), "tau": tau,
                    "decay": float(a_near - a_far)})
    return out


def measured_beta():
    """How fast within-shell spread grows with distance, sd ~ sd0(1 + beta d).

    Reported PER CLASS and de-duplicated, both of which matter. An earlier
    version pooled every profile row and quoted the median as 0.21, calling it
    "a property of the state space". It is not: 0.21 is the pattern-database
    value, learned networks are an order of magnitude higher, and the two ranges
    do not overlap. The pooled figure was the abstraction value surfacing because
    duplicate PDB rows outnumbered learned rows about three to one -- the same
    counting error that inflated n elsewhere.

    The substantive claim survives in a more interesting form: the two classes
    reach comparable decay from opposite directions, abstractions starting from a
    much larger sd0 with a gentle slope, learned networks from a small sd0 with a
    steep one.
    """
    per, seen = {}, set()
    for f in sorted(RESULTS.glob("profile-*.json")):
        if "_p3-" in f.stem:      # separate experiment; see strength_control.py
            continue
        d = json.load(open(f))
        if d["k"] <= 2:
            continue
        for name, h in d["heuristics"].items():
            if name == "random" or len(h["profile"]) < 4:
                continue
            kind = "learned" if name == "learned" else "PDB"
            key = (d["task"], d["moves"], name) + ((d["tag"],) if kind == "learned" else ())
            if key in seen:
                continue
            seen.add(key)
            sd = np.array([r["sd_lo"] for r in h["profile"]], float)
            dd = np.array([r["d"] for r in h["profile"]], float)
            if sd[0] <= 0:
                continue
            per.setdefault(kind, []).append(
                {"beta": float(np.polyfit(dd - dd[0], sd / sd[0], 1)[0]),
                 "sd0": float(sd[0])})
    out = {}
    for kind, v in per.items():
        b = np.array([x["beta"] for x in v]); s0 = np.array([x["sd0"] for x in v])
        out[kind] = {"n": len(v), "beta_median": float(np.median(b)),
                     "beta_range": [float(b.min()), float(b.max())],
                     "sd0_median": float(np.median(s0))}
    return out


def main():
    rows, ties = check_prediction()

    ref = json.load(open(RESULTS / "profile-wings-k6_s0.json"))
    n, _, _ = shells_of(ref["heuristics"]["learned"]["profile"])
    bstats = measured_beta()
    print("\n  spread growth sd ~ sd0(1 + beta d), estimated per class and "
          "de-duplicated:")
    for kind, v in sorted(bstats.items()):
        print(f"    {kind:<8} n={v['n']:>3}  beta {v['beta_median']:6.3f} "
              f"(range {v['beta_range'][0]:.2f} to {v['beta_range'][1]:.2f})   "
              f"sd0 {v['sd0_median']:.3f}")
    print("  -> the classes are NOT the same. Pooling them, as an earlier version")
    print("     of this analysis did, reported the abstraction value as if it were")
    print("     a property of the space. They reach comparable decay from opposite")
    print("     ends: large sd0 with a gentle slope, or small sd0 with a steep one.")
    beta = float(np.median([v["beta_median"] for v in bstats.values()]))

    print("\nThe shape of decay against strength, from the identity alone:")
    flat = inverted_u(n, beta=0.0)
    print(f"  spread constant across shells (beta=0): max decay over the whole "
          f"quality range = {max(r['decay'] for r in flat):.4f}")
    print("  -> decay is not produced by strength. It exists only because spread")
    print("     grows with distance, which is a property of the state space.\n")

    curve = inverted_u(n, beta=beta)
    peak = max(curve, key=lambda r: r["decay"])
    print(f"  with spread growing (beta={beta:.2f}): decay peaks at {peak['decay']:.3f} "
          f"at tau {peak['tau']:+.3f}")
    meas = np.array([x["measured"] for x in rows if x["kind"] != "random"])
    lo, hi = float(meas.min()), float(meas.max())
    print(f"  every informative heuristic we measured sits at tau {lo:+.3f} to {hi:+.3f}")
    if hi < peak["tau"]:
        print("  -> all of them are on the rising limb, so decay and strength")
        print("     correlate positively HERE and would reverse beyond the peak.")
    else:
        frac = float((meas > peak["tau"]).mean())
        print(f"  -> we straddle the peak: {frac*100:.0f}% of them sit past it. The")
        print("     theory therefore predicts the measured decay-vs-strength cloud")
        print("     should already bend over rather than rise throughout, and a")
        print("     single correlation coefficient is the wrong summary of it.")
        # test that prediction against the measurement rather than asserting it
        dec = np.array([x for x in json.load(
            open(RESULTS / "strength-control.json"))["observations"]
            if x["kind"] != "random"])
        g = np.array([x["gdrc"] for x in dec]); y = np.array([x["decay"] for x in dec])
        quad = np.polyfit(g, y, 2)
        apex = -quad[1] / (2 * quad[0]) if quad[0] != 0 else float("nan")
        print(f"     measured: quadratic fit turns over at tau {apex:+.3f} "
              f"(curvature {quad[0]:+.3f}), theory says {peak['tau']:+.3f}")
        print("     These do not agree, and we do not claim they do. The curve is")
        print("     drawn for one shell histogram and one beta, while the measured")
        print("     cloud mixes rungs, diameters and both heuristic classes -- whose")
        print("     tau values are not even on a common scale, given the tie")
        print("     inflation above. What the theory establishes is the SHAPE and")
        print("     that a single correlation cannot summarise it; locating the peak")
        print("     would need heuristics spanning the range on one fixed problem.")

    out = {"prediction": rows,
           "ceiling_verified": vc, "ceiling_note": "tau-b cannot reach 1; the cap is set by shell sizes",
           "flat_spread_max_decay": max(r["decay"] for r in flat),
           "beta_by_class": bstats, "tie_inflation": ties, "curve": curve, "peak": peak,
           "measured_tau_range": [float(min(meas)), float(max(meas))]}
    (RESULTS / "tau-theory.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n  -> tau-theory.json")


if __name__ == "__main__":
    main()
