"""Does the two-moment law survive being deliberately broken?

Everything in dprime_law.py is observational: we measured heuristics that
happened to exist and found Phi(gap / (sd * sqrt2)) described them. Two
objections follow, and neither is answerable from that evidence.

  1. It might be correlational. Heuristics differ in rung, diameter, network and
     training, so any of those could drive both moments and accuracy.
  2. It might be vacuous. ANY increasing function of gap/sd would track the
     measurement, so agreement does not show that the GAUSSIAN form is doing the
     work rather than mere monotonicity.

This script answers both by manufacturing heuristics with known perturbations.
Take a trained network and add noise: h_lambda(s) = h(s) + lambda * eps(s). Rung,
diameter, network and training are then identical across the whole sweep by
construction, so objection 1 disappears. And the noise family is ours to choose:

  gaussian   the law's own assumption -- should hold
  uniform    light-tailed, still finite variance -- should hold
  laplace    heavy-tailed, finite variance -- should hold, slightly worse
  cauchy     NO FINITE VARIANCE -- the law has no right to work here

Cauchy is the point of the exercise. A within-shell "spread" estimated from
Cauchy draws does not converge to anything, so if Phi still predicts accuracy
under Cauchy noise then the fit was never about the normal distribution and the
paper's central claim is only that accuracy increases with gap/sd. That would be
a much weaker claim than the one we make, and we would rather discover it here.

The noise is DETERMINISTIC per state, derived by hashing the state's index in the
exact table. A heuristic that returns a different value each time it is called is
not a heuristic, and beam search over one would not be well defined.

    ../.venv-mlx/bin/python noise_dose.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from davi import Task
from evaluate import j_of, load
from exact import indexer
from profiles import sweep_sample

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"
SQ2 = np.sqrt(2.0)

# variance of each family at unit scale; None where it does not exist
FAMILIES = {"gaussian": 1.0, "uniform": 1.0 / 3.0, "laplace": 2.0, "cauchy": None}


def deterministic_noise(idx, family):
    """A fixed pseudo-random draw per state, from its exact-table index.

    Hashing the index rather than drawing from a stream keeps h a function of the
    state, which is what makes the perturbed object a heuristic at all.
    """
    # splitmix64-style avalanche, then map to (0,1)
    z = (idx.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    if family == "gaussian":
        return norm.ppf(u)
    if family == "uniform":
        return u * 2.0 - 1.0
    if family == "laplace":
        return np.where(u < 0.5, np.log(2 * u), -np.log(2 * (1 - u))) / np.sqrt(2.0)
    if family == "cauchy":
        return np.tan(np.pi * (u - 0.5))
    raise ValueError(family)


def profile_split(h, d, shells, half, rng, pairs=20000):
    """Moments from half A, accuracy from half B. Same protocol as profiles.py."""
    out = []
    for v in shells:
        lo_a, hi_a = h[(d == v) & half], h[(d == v + 1) & half]
        lo_b, hi_b = h[(d == v) & ~half], h[(d == v + 1) & ~half]
        if min(lo_a.size, hi_a.size, lo_b.size, hi_b.size) < 50:
            continue
        sd = np.sqrt((lo_a.std() ** 2 + hi_a.std() ** 2) / 2.0)
        if sd <= 0:
            continue
        m = min(pairs, lo_b.size * hi_b.size)
        ia, ib = rng.integers(0, lo_b.size, m), rng.integers(0, hi_b.size, m)
        acc = float(np.mean(lo_b[ia] < hi_b[ib]) + 0.5 * np.mean(lo_b[ia] == hi_b[ib]))
        gap = float(hi_a.mean() - lo_a.mean())
        out.append({"d": int(v), "acc": acc, "gap": gap, "sd": float(sd),
                    "pred": float(norm.cdf(gap / sd / SQ2))})
    return out


def main():
    task, net, meta = load("wings-k6", "_s0")
    rng = np.random.default_rng(7)
    st = sweep_sample(task, 2500, 16, rng)
    idx = indexer(task.k)(st)
    d = np.load(HERE / "exact_k6.npy")[idx].astype(np.int64)
    h0 = j_of(task, net, st).astype(np.float64)

    uniq, cnt = np.unique(d, return_counts=True)
    shells = [int(v) for v, c in zip(uniq, cnt) if v > 0 and c >= 200][:-1]
    half = np.zeros(st.shape[0], dtype=bool)
    half[::2] = True

    # base within-shell spread, so lambda is in units the heuristic understands
    base_sd = float(np.median([h0[d == v].std() for v in shells]))
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"wings-k6 s0: {st.shape[0]:,} states, shells {shells[0]}..{shells[-1]+1}, "
          f"median within-shell spread {base_sd:.4f}\n")
    print(f"  {'family':<10}{'lambda':>7}{'n':>4}{'measured MAE':>14}"
          f"{'forward MAE':>13}   (forward = predicted from the lambda=0 moments)")

    rows = []
    for fam, var in FAMILIES.items():
        for lam in lams:
            if fam != "gaussian" and lam == 0.0:
                continue                       # identical to gaussian at lambda 0
            h = h0 + lam * base_sd * deterministic_noise(idx, fam)
            prof = profile_split(h, d, shells, half, np.random.default_rng(11))
            if not prof:
                continue
            a = np.array([r["acc"] for r in prof])
            p = np.array([r["pred"] for r in prof])
            mae = float(np.abs(a - p).mean())

            # FORWARD prediction: no moments from the perturbed heuristic at all.
            # Adding independent noise of variance v inflates each shell's
            # variance by exactly lam^2 * base_sd^2 * v and leaves the gap alone,
            # so the law makes a genuine out-of-sample prediction -- except for
            # cauchy, where v does not exist and no prediction can be made.
            fmae = None
            if var is not None:
                base = profile_split(h0, d, shells, half, np.random.default_rng(11))
                bs = {r["d"]: r for r in base}
                fp, fa = [], []
                for r in prof:
                    b = bs.get(r["d"])
                    if not b:
                        continue
                    sd2 = np.sqrt(b["sd"] ** 2 + (lam * base_sd) ** 2 * var)
                    fp.append(float(norm.cdf(b["gap"] / sd2 / SQ2)))
                    fa.append(r["acc"])
                fmae = float(np.abs(np.array(fa) - np.array(fp)).mean())

            rows.append({"family": fam, "lam": lam, "n": len(prof), "mae": mae,
                         "forward_mae": fmae,
                         "mean_acc": float(a.mean()), "mean_pred": float(p.mean())})
            print(f"  {fam:<10}{lam:>7.2f}{len(prof):>4}{mae:>14.4f}"
                  + (f"{fmae:>13.4f}" if fmae is not None else f"{'n/a':>13}"))

    g = [r for r in rows if r["family"] == "gaussian" and r["lam"] > 0]
    c = [r for r in rows if r["family"] == "cauchy"]
    print(f"\n  gaussian, lambda>0 : measured MAE {np.mean([r['mae'] for r in g]):.4f}   "
          f"forward MAE {np.mean([r['forward_mae'] for r in g]):.4f}")
    print(f"  cauchy             : measured MAE {np.mean([r['mae'] for r in c]):.4f}   "
          f"forward MAE  n/a (no finite variance to forward-predict from)")
    verdict = ("the Gaussian form is doing the work"
               if np.mean([r["mae"] for r in c]) > 3 * np.mean([r["mae"] for r in g])
               else "WARNING: the law survives Cauchy noise, so it is only "
                    "monotonicity in gap/sd, not the normal form")
    print(f"  -> {verdict}")

    (RESULTS / "noise-dose.json").write_text(json.dumps(
        {"task": "wings-k6_s0", "base_sd": base_sd, "shells": shells,
         "lambdas": lams, "families": {k: v for k, v in FAMILIES.items()},
         "rows": rows, "verdict": verdict}, indent=2) + "\n")
    print("\n  -> noise-dose.json")


if __name__ == "__main__":
    main()
