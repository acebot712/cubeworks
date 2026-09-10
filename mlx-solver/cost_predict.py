"""Does the profile predict search cost better than pooled GDRC?

Everything about this experiment is fixed in
`eval/PREREGISTRATION-search-cost.md`, written and committed before any of it
ran. That document, not this file, is the authority on the predictor, the
outcome, the test and the threshold. Deviations go in its Deviations section.

Superseded: `e5_predict.py` asked the same question against required beam width
and did not survive its own censoring. It is kept for the record.

    ../.venv-mlx/bin/python cost_predict.py --selftest
    ../.venv-mlx/bin/python cost_predict.py --run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from domains import load_table, make_task, rungs
from evaluate import j_of, load
from profiles import sweep_sample
from resolution import sample_states
from search_cost import greedy_expansions, pdb_heuristic, random_heuristic

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"

# Pre-registered. Six tasks, each analysed on its own; the restricted movesets
# are different state graphs with their own exact tables.
TASKS = [("tile-2x4", "all"), ("tile-3x3", "all"),
         ("wings-k4", "all"), ("wings-k4", "oi-q"),
         ("wings-k4", "oi-q3"), ("wings-k4", "oi-q4")]
N_INSTANCES = 1000
WALK = 500          # verified against the BFS histogram; see the pre-registration
BOOTSTRAP = 10000


def shell_pop(task):
    """True population of each shell, from the exhaustive BFS histogram.

    The predictor weights adjacent-shell accuracy by how many pairs the state
    space actually contains, so the weights come from the whole space and not
    from whatever the profile happened to sample.
    """
    d = json.load(open(task.histogram_path()))
    pops = np.zeros(max(r["depth"] for r in d["histogram"]) + 1, dtype=np.float64)
    for r in d["histogram"]:
        pops[r["depth"]] = r["count"]
    return pops


def measure(h, true_d, pops, seed=101, pairs=20000, min_shell=200):
    """The predictor and pooled GDRC for one heuristic, from one sample.

    PREDICTOR: mean adjacent-shell ordering accuracy, weighted by n_d * n_{d+1}.
    GDRC: Kendall tau-b over every pair, which is Wilt and Ruml's statistic.

    The contrast is the whole point and it is not tautological only because of
    the restriction: tau-b is a shell-size weighted sum over ALL pairs, and the
    predictor is the same weighting over ADJACENT pairs alone. Adjacent-only
    versus all-pairs is the dilution argument.
    """
    # A FRESH rng at a fixed seed, not a shared advancing one. Every heuristic
    # is then scored on the identical sampled pairs, so the difference between
    # two predictors carries no sampling noise of its own. On wings-k4/all
    # sixteen heuristics span a predictor range of 0.679 to 0.936, and
    # independent binomial error on each estimate is a real fraction of that.
    rng = np.random.default_rng(seed)
    gdrc = float(kendalltau(h, true_d).statistic)
    num = den = 0.0
    for d in range(1, len(pops) - 1):
        a, b = h[true_d == d], h[true_d == d + 1]
        if len(a) < min_shell or len(b) < min_shell:
            continue
        m = min(pairs, len(a) * len(b))
        ia, ib = rng.integers(0, len(a), m), rng.integers(0, len(b), m)
        acc = float(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
        w = pops[d] * pops[d + 1]
        num += w * acc
        den += w
    return {"gdrc": gdrc, "predictor": float(num / den) if den else float("nan")}


def population(task):
    """Every heuristic for this task: each rung, each final learned checkpoint.

    Intermediate `@step` snapshots are excluded: they are not independent of the
    final checkpoint of the run they came from.
    """
    out = []
    for j in rungs(task):
        out.append((f"PDB(k={j})", "PDB", pdb_heuristic(task, j)))
    for meta_f in sorted(HERE.glob(f"ckpt_{task.name}*.json")):
        stem = meta_f.stem[len("ckpt_"):]
        if "@" in stem or stem.startswith("wings-k4_p3-"):
            continue                       # snapshot, or a different experiment
        meta = json.loads(meta_f.read_text())
        if meta.get("moves", "all") != task.moveset:
            continue
        tag = stem[len(task.name):]
        try:
            t2, net, _ = load(task.name, tag)
        except Exception as exc:
            # Loud, because silently shrinking the population changes n and the
            # predictor spread the whole result depends on, with no trace in the
            # output that anything was lost.
            print(f"    WARNING: checkpoint {stem} failed to load and is NOT in "
                  f"the population: {exc}", flush=True)
            continue
        out.append((f"learned{tag}", "learned",
                    (lambda n=net, t=t2: (lambda s: j_of(t, n, s)))()))
    out.append(("random", "random", random_heuristic(0)))
    return out


def run_task(name, moves, n_inst, seed, per_len=2500, max_len=45):
    task = make_task(name, moves=moves)
    exact = load_table(task)
    pops = shell_pop(task)
    rng = np.random.default_rng(seed)

    # Two independent samples. Instances are uniform, by deep random walk;
    # the profile sample sweeps scramble length so every shell is populated.
    # Reusing one sample for both would correlate predictor with outcome.
    starts = sample_states(task, n_inst, WALK, rng)
    prof_states = sweep_sample(task, per_len, max_len, rng)
    prof_d = exact[task.rank(prof_states)].astype(np.int32)

    rows = []
    for label, kind, h_fn in population(task):
        m = measure(h_fn(prof_states), prof_d, pops)
        e, L = greedy_expansions(task, h_fn, starts, chunk=250)
        opt = exact[task.rank(starts)].astype(np.int64)
        # L is -1 where a search hit the cap. Averaging that in would report a
        # NEGATIVE suboptimality, which no real solution can have.
        solved = L >= 0
        rows.append({"task": f"{name}/{moves}", "name": label, "kind": kind,
                     "gdrc": m["gdrc"], "predictor": m["predictor"],
                     "median_expansions": float(np.median(e)),
                     "geomean_expansions": float(np.exp(np.mean(np.log(np.maximum(e, 1))))),
                     "mean_suboptimality": (float(np.mean((L - opt)[solved]))
                                            if solved.any() else float("nan")),
                     "n_unsolved": int((L < 0).sum())})
        print(f"    {label:<16} {kind:<8} predictor {m['predictor']:.4f}  "
              f"GDRC {m['gdrc']:+.4f}  median expansions "
              f"{rows[-1]['median_expansions']:>9,.0f}", flush=True)
    return rows


def spearman_gap(rows, outcome="median_expansions"):
    """Spearman(predictor, cost) minus Spearman(GDRC, cost).

    The difference, not either correlation alone: both are high simply because
    both track heuristic strength, so only the gap between them carries the
    claim. Cost rises as quality falls, so a better predictor is a MORE negative
    correlation; both are negated so that positive means better.
    """
    y = np.array([r[outcome] for r in rows])
    p = np.array([r["predictor"] for r in rows])
    g = np.array([r["gdrc"] for r in rows])
    rp = -spearmanr(p, y).statistic
    rg = -spearmanr(g, y).statistic
    return float(rp - rg), float(rp), float(rg)


def bootstrap_gap(rows, outcome="median_expansions", reps=BOOTSTRAP, seed=11):
    """Paired bootstrap over heuristics, which is where the uncertainty lives."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    out, degenerate = [], 0
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sub = [rows[i] for i in idx]
        d, _, _ = spearman_gap(sub, outcome)
        if not np.isfinite(d):
            # Neither correlation is estimable in this resample, so there is no
            # evidence of a difference in it. Recording 0 keeps the interval
            # honest; discarding it would condition the CI on non-degeneracy and
            # report [+0.000, +0.000] as if it were a precise estimate.
            d, degenerate = 0.0, degenerate + 1
        out.append(d)
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), degenerate


def combined_bootstrap(by_task, outcome="median_expansions", reps=BOOTSTRAP, seed=13):
    """Resample heuristics WITHIN each task, average the per-task gaps.

    Resampling inside the task is what keeps the comparison paired and stops a
    between-task difference, branching factor above all, entering as if it were
    a predictor effect.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        gaps = []
        for rows in by_task.values():
            idx = rng.integers(0, len(rows), len(rows))
            sub = [rows[i] for i in idx]
            d, _, _ = spearman_gap(sub, outcome)
            # Every task contributes to every draw. Skipping the degenerate ones
            # made the denominator vary, and the tasks that dropped out were the
            # zero-gap ones, so the combined mean came out inflated: +0.0855
            # against a plain task-gap mean of +0.0751 on the median outcome.
            gaps.append(d if np.isfinite(d) else 0.0)
        draws.append(float(np.mean(gaps)))
    a = np.array(draws)
    return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "reps": len(a)}


def analyse(by_task):
    """The pre-registered test on the primary outcome, and its declared check.

    Both are reported because the primary turned out to quantise: a median over
    1000 instances is a small integer on these tasks, so 12 heuristics can land
    on 3 distinct costs and a rank test has almost no resolution left. The
    geometric mean was pre-registered as a robustness check precisely so that
    this could be looked at without it being a new choice made after the fact.
    """
    res = {}
    for outcome, role in (("median_expansions", "primary"),
                          ("geomean_expansions", "robustness check")):
        print(f"\n\n=========== PRE-REGISTERED ANALYSIS: {outcome} ({role}) ===========")
        per_task, positive = {}, 0
        for key, rows in by_task.items():
            reg = [r for r in rows if r["kind"] != "random"]  # anchor, not a member
            y = [r[outcome] for r in reg]
            gap, rp, rg = spearman_gap(reg, outcome)
            lo, hi, degen = bootstrap_gap(reg, outcome)
            per_task[key] = {"n": len(reg), "distinct_costs": len(set(y)),
                             "gap": gap, "r_predictor": rp, "r_gdrc": rg,
                             "ci95": [lo, hi], "degenerate_resamples": degen,
                             "evaluable": bool(np.isfinite(gap))}
            positive += bool(np.isfinite(gap)) and gap > 0
            print(f"  {key:<16} n={len(reg):>3} distinct={len(set(y)):>3}  "
                  f"predictor {rp:+.3f}  GDRC {rg:+.3f}  gap {gap:+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]"
                  + (f"  ({degen:,} degenerate resamples)" if degen else ""))
        comb = combined_bootstrap({k: [r for r in v if r["kind"] != "random"]
                                   for k, v in by_task.items()}, outcome)
        evaluable = sum(v["evaluable"] for v in per_task.values())
        # The 4-of-6 threshold is pre-registered and stays as written. Reporting
        # how many tasks were evaluable at all is what stops a task that could
        # not be measured from silently reading as evidence against.
        success = positive >= 4 and comb["lo"] > 0
        print(f"\n  combined gap {comb['mean']:+.4f}  95% CI "
              f"[{comb['lo']:+.4f}, {comb['hi']:+.4f}]")
        print(f"  positive in {positive} of {len(per_task)} tasks (need 4); "
              f"{evaluable} of {len(per_task)} were evaluable")
        print(f"  RESULT: {'SUCCESS' if success else 'NULL'}")
        res[outcome] = {"role": role, "per_task": per_task, "combined": comb,
                        "tasks_positive": int(positive), "n_tasks": len(per_task),
                        "tasks_evaluable": int(evaluable),
                        "success": bool(success)}
    return res


# ------------------------------------------------------------------- selftest
def selftest():
    """Checks at the two seams: the predictor, and the statistic."""
    ok = True
    task = make_task("tile-3x3")
    exact = load_table(task)
    pops = shell_pop(task)
    rng = np.random.default_rng(0)
    st = sweep_sample(task, 1500, 45, rng)
    d = exact[task.rank(st)].astype(np.int32)

    # SEAM 1, the predictor. A heuristic that IS the distance orders every
    # adjacent pair correctly, so the weighted accuracy is 1 and tau-b is 1.
    m = measure(exact[task.rank(st)].astype(np.float64), d, pops)
    perfect = abs(m["predictor"] - 1.0) < 1e-9 and abs(m["gdrc"] - 1.0) < 1e-9
    print(f"  oracle   predictor {m['predictor']:.4f} (want 1)   "
          f"GDRC {m['gdrc']:+.4f} (want +1)   {'OK' if perfect else '*** FAIL ***'}")
    ok &= perfect

    m = measure(rng.random(st.shape[0]), d, pops)
    chance = abs(m["predictor"] - 0.5) < 0.02 and abs(m["gdrc"]) < 0.02
    print(f"  random   predictor {m['predictor']:.4f} (want 0.5) "
          f"  GDRC {m['gdrc']:+.4f} (want 0)   {'OK' if chance else '*** FAIL ***'}")
    ok &= chance

    # The weighting has to bite: population weights and uniform weights differ
    # whenever shells differ in size, and the pre-registered predictor is the
    # population-weighted one.
    flat = np.ones_like(pops)
    j = rungs(task)[1]
    hv = pdb_heuristic(task, j)(st)
    a = measure(hv, d, pops)["predictor"]
    b = measure(hv, d, flat)["predictor"]
    differs = abs(a - b) > 1e-6
    print(f"  weighting population {a:.4f} vs uniform {b:.4f}   "
          f"{'OK, weights bite' if differs else '*** FAIL, weighting is inert ***'}")
    ok &= differs

    # SEAM 2, the statistic. Build rows where the predictor ranks cost perfectly
    # and GDRC ranks it backwards; the gap must come out strongly positive.
    good = [{"predictor": 0.5 + 0.05 * i, "gdrc": 0.9 - 0.05 * i,
             "median_expansions": 1000.0 - 100 * i} for i in range(8)]
    gap, rp, rg = spearman_gap(good)
    right = gap > 1.5 and rp > 0.99 and rg < -0.99
    print(f"  statistic predictor beats GDRC: gap {gap:+.3f} "
          f"(predictor {rp:+.3f}, GDRC {rg:+.3f})   {'OK' if right else '*** FAIL ***'}")
    ok &= right

    same = [{"predictor": p, "gdrc": p, "median_expansions": 1000.0 - 100 * i}
            for i, p in enumerate(np.linspace(0.5, 0.9, 8))]
    gap0, _, _ = spearman_gap(same)
    null = abs(gap0) < 1e-9
    print(f"  statistic identical predictors: gap {gap0:+.3f} (want 0)   "
          f"{'OK' if null else '*** FAIL ***'}")
    ok &= null

    lo, hi, degen = bootstrap_gap(good, reps=2000)
    excludes = lo > 0 and degen == 0
    print(f"  bootstrap 95% CI on that gap [{lo:+.3f}, {hi:+.3f}], "
          f"{degen} degenerate resamples   "
          f"{'OK, excludes zero' if excludes else '*** FAIL ***'}")
    ok &= excludes

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse-only", action="store_true",
                    help="redo the analysis from the saved rows, no searches")
    ap.add_argument("--instances", type=int, default=N_INSTANCES)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.analyse_only:
        saved = json.load(open(RESULTS / "cost-predict.json"))
        by = {}
        for r in saved["rows"]:
            by.setdefault(r["task"], []).append(r)
        saved |= analyse(by)
        (RESULTS / "cost-predict.json").write_text(json.dumps(saved, indent=2) + "\n")
        print("\n  -> cost-predict.json")
        raise SystemExit(0)
    if not args.run:
        raise SystemExit("give --selftest, --run or --analyse-only")

    by_task = {}
    for name, moves in TASKS:
        key = f"{name}/{moves}"
        print(f"\n=== {key} ===", flush=True)
        by_task[key] = run_task(name, moves, args.instances, args.seed)

    out = {"preregistration": "eval/PREREGISTRATION-search-cost.md",
           "predictor": "adjacent-shell accuracy weighted by n_d*n_(d+1)",
           "n_instances": args.instances, "seed": args.seed,
           "rows": [r for v in by_task.values() for r in v]}
    out |= analyse(by_task)
    (RESULTS / "cost-predict.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  -> cost-predict.json")


if __name__ == "__main__":
    main()
