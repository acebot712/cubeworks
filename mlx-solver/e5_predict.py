"""E5: does an intrinsic property of a heuristic predict the search it needs?

For every configuration where exact ground truth exists, pair two measurements
that are made independently of each other:

  from resolution/profiles   ranking accuracy per true-distance shell, and in
                             particular the accuracy near the problem's diameter: computed WITHOUT running any search
  from budget_curve          the smallest beam width reaching a target solve
                             rate, the search cost, with the algorithm's own
                             budget divided out

If the first predicts the second, a trained heuristic can be graded cheaply and
its search cost forecast before a search is ever run.

Two things to be honest about while reading the output.

Wilt & Ruml (JAIR 2016) already showed a pooled Kendall tau against exact
distance predicts greedy-best-first expansions, so "a ranking statistic predicts
search cost" is not itself new. What is being tested here is whether the
PER-SHELL value near the diameter beats the pooled one, which is the only part
that could be ours.

And Korf-Reid-Edelkamp implies search cost depends on branching factor and
solution depth, not on a heuristic statistic alone. Our configurations vary
branching (63 down to 12) and depth (6 to 12) together, so a correlation here is
NOT evidence that state-space size is irrelevant, and must not be reported as
such. The honest reading of a positive result is "profile is a better predictor
than pooled tau on this family", nothing larger.

    ../.venv-mlx/bin/python e5_predict.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

HERE = Path(__file__).parent
PY = str(Path(sys.executable))
RESULTS = HERE.parent / "eval" / "results"

# (task, tag) pairs that have both a checkpoint and an exact distance table
CONFIGS = [
    ("wings-k2", "_s0@3000"), ("wings-k2", "_s0@12000"), ("wings-k2", "_s0"),
    ("wings-k4", "_s0@3000"), ("wings-k4", "_s0@12000"), ("wings-k4", "_s0"),
    ("wings-k6", "_s0@3000"), ("wings-k6", "_s0@12000"), ("wings-k6", "_s0"),
    ("wings-k4", "_dv-all_s0"), ("wings-k4", "_dv-all_s1"), ("wings-k4", "_dv-all_s2"),
    ("wings-k4", "_dv-oi-q_s0"), ("wings-k4", "_dv-oi-q_s1"), ("wings-k4", "_dv-oi-q_s2"),
    ("wings-k4", "_dv-oi-q4_s0"), ("wings-k4", "_dv-oi-q4_s1"),
    ("wings-k4", "_dv-oi-q3_s0"), ("wings-k4", "_dv-oi-q3_s1"), ("wings-k4", "_dv-oi-q3_s2"),
]


def ensure(script, task, tag, extra, out_name, force=False):
    out = RESULTS / out_name
    if out.exists() and not force:
        return json.loads(out.read_text())
    cmd = [PY, str(HERE / script), "--task", task, "--tag", tag] + extra
    log = HERE / f"log_e5_{script[:4]}_{task}{tag}.txt".replace("@", "at")
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return json.loads(out.read_text()) if out.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-budget", type=int, default=100)
    ap.add_argument("--n-profile", type=int, default=30000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = []
    for task, tag in CONFIGS:
        if not (HERE / f"ckpt_{task}{tag}.json").exists():
            print(f"  skip {task}{tag}: no checkpoint", flush=True)
            continue
        prof = ensure("profiles.py", task, tag,
                      ["--pdb-k", "2", "--n", str(args.n_profile)],
                      f"profile-{task}{tag}.json", args.force)
        bud = ensure("budget_curve.py", task, tag,
                     ["--n", str(args.n_budget)],
                     f"budget-{task}{tag}.json", args.force)
        if not prof or not bud:
            print(f"  skip {task}{tag}: measurement failed", flush=True)
            continue

        p = prof["heuristics"]["learned"]["profile"]
        if not p:
            continue
        accs = [r["acc"] for r in p]
        rows.append({
            "task": task, "tag": tag, "moves": prof["moves"], "k": prof["k"],
            "states": prof["states"], "step": prof["step"],
            "diameter": max(r["d"] for r in p) + 1,
            "gdrc": prof["heuristics"]["learned"]["gdrc"],
            "acc_deepest": accs[-1],
            "acc_min": min(accs),
            "acc_mean": float(np.mean(accs)),
            "w50": bud["w50"], "w90": bud["w90"],
        })
        r = rows[-1]
        print(f"  {task}{tag:<16} diam {r['diameter']:>2}  GDRC {r['gdrc']:+.3f}  "
              f"acc_min {r['acc_min']:.3f}  w50 "
              f"{(f'{r['w50']:.1f}' if r['w50'] else 'n/a'):>6}  w90 "
              f"{(f'{r['w90']:.1f}' if r['w90'] else 'n/a'):>6}", flush=True)

    ok = [r for r in rows if r["w90"] is not None]
    print(f"\n{len(rows)} configs measured, {len(ok)} reached 90% solve\n")
    if len(ok) < 4:
        print("too few to correlate: need more configs that discriminate")
        return

    y = np.log([r["w90"] for r in ok])
    print("Predicting log(beam width for 90% solve):")
    print(f"  {'predictor':>14} {'Pearson r':>11} {'Spearman':>10} {'Kendall':>9}")
    for name in ("gdrc", "acc_deepest", "acc_min", "acc_mean"):
        x = np.array([r[name] for r in ok])
        if np.std(x) < 1e-9:
            continue
        print(f"  {name:>14} {pearsonr(x, y).statistic:>11.3f} "
              f"{spearmanr(x, y).statistic:>10.3f} {kendalltau(x, y).statistic:>9.3f}")

    out = RESULTS / "e5-prediction.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"\n  -> {out.name}")


if __name__ == "__main__":
    main()
