"""Ticket 01: what do the corpus defects cost the numbers already in the paper?

Three defects live in `tau_theory.py`. One of its two corpus loops does not
deduplicate. The other deduplicates but never splits by domain. A third site
fits one curve across both domains. This script measures what each of them
actually moves, so that a later refactor can be told apart from a correction.

It changes no committed script. It replicates the affected analyses here, with
the corrections applied, and diffs against the committed baseline the paper's
numbers came from.

ONE THING HAD TO BE REPAIRED TO MEASURE ANYTHING. `tau_theory.py` does not run.
Its tie-inflation analysis imports `project` from `profiles`, which was removed
when both domains were put behind one interface, so the script has raised a
TypeError for every commit since. The repair here is the single call, made
against `domains.project`; nothing else about the analysis is touched.

    ../.venv-mlx/bin/python defect_cost.py
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr

from domains import domain_of_name, make_task
from exact import indexer
from profiles import sweep_sample
from tau_theory import shells_of, tau_from_profile

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"

# The commit whose tau-theory.json the paper's numbers were taken from. The SHA
# alone is not a durable record: this repository's history has been rewritten once
# already, when the camera frames were purged, and every SHA changed with it. The
# values are therefore recorded here too, so that a future rewrite costs the
# comparison its provenance but not its content.
BASELINE_REV = "ac3a246"
BASELINE_RECORDED = {"tie_inflation_mean": 0.0843, "tie_inflation_max": 0.1012,
                     "beta_learned": 2.043, "beta_pdb": 0.149}

# The tie-inflation analysis covers these and only these. All cube, hardcoded,
# and it reads no corpus, which is the reason neither defect can reach it.
TIE_CONFIGS = (("wings-k4", "oi-q3", 4), ("wings-k4", "all", 4), ("wings-k6", "all", 6))

# Decimal places for both the printed difference and the moved flag.
MOVED_DP = 6


def baseline():
    out = subprocess.run(["git", "-C", str(HERE.parent), "show",
                          f"{BASELINE_REV}:eval/results/tau-theory.json"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cannot read the baseline at {BASELINE_REV}")
    return json.loads(out.stdout)


def tie_inflation():
    """The abstract's figure. Identical to the committed analysis but runnable."""
    rows = []
    for tname, mv, k in TIE_CONFIGS:
        suf = "" if mv == "all" else f"-{mv}"
        if not (HERE / f"exact_k{k}{suf}.npy").exists():
            continue
        task = make_task(tname, moves=mv)
        st = sweep_sample(task, 2500, 16, np.random.default_rng(7))
        d = np.load(HERE / f"exact_k{k}{suf}.npy")[indexer(k)(st)].astype(np.int64)
        rng = np.random.default_rng(3)
        for j in range(2, k):
            p = HERE / f"exact_k{j}{suf}.npy"
            if not p.exists():
                continue
            h = np.load(p)[indexer(j)(task.project(st, j))].astype(np.float64)
            tied = float(kendalltau(h, d).statistic)
            free = float(kendalltau(h + rng.random(h.size) * 0.999, d).statistic)
            rows.append({"task": tname, "moves": mv, "pdb_k": j,
                         "inflation": tied - free})
    return rows


def corpus(dedup, min_shells=2, want_moments=False):
    """The corpus loop, with deduplication switchable so its cost is measurable.

    Yes, this is a fifth copy of the loop the tickets exist to remove. It is
    deliberate and temporary: measuring what deduplication costs requires being
    able to turn it off, which the corpus module will rightly refuse to allow.
    Delete this file once ticket 03 has landed and its numbers are recorded.
    """
    rows, seen = [], set()
    for f in sorted(RESULTS.glob("profile-*.json")):
        if "_p3-" in f.stem:
            continue
        d = json.load(open(f))
        if d["k"] <= 2:
            continue
        for name, h in d["heuristics"].items():
            prof = h["profile"]
            if len(prof) < min_shells or (not want_moments and "n_lo" not in prof[0]):
                continue
            if want_moments and name == "random":
                continue
            kind = ("learned" if name == "learned"
                    else "random" if name == "random" else "PDB")
            if dedup:
                key = ((d["task"], d["moves"], name)
                       + ((d["tag"],) if kind == "learned" else ()))
                if key in seen:
                    continue
                seen.add(key)
            row = {"kind": kind, "domain": domain_of_name(d["task"]), "name": name}
            if want_moments:
                sd = np.array([r["sd_lo"] for r in prof], float)
                dd = np.array([r["d"] for r in prof], float)
                if sd[0] <= 0:
                    continue
                # exactly the arithmetic measured_beta uses
                row["beta"] = float(np.polyfit(dd - dd[0], sd / sd[0], 1)[0])
            else:
                n, mu, sd = shells_of(prof)
                row |= {"measured": h["gdrc"], "predicted": tau_from_profile(n, mu, sd)}
            rows.append(row)
    return rows


def report_prediction(rows, label):
    if not rows:
        print(f"  {label}  no profiles found; has any measurement been run?")
        return
    m = np.array([r["measured"] for r in rows])
    p = np.array([r["predicted"] for r in rows])
    print(f"  {label}  n={len(rows):>3}  {dict(Counter(r['kind'] for r in rows))}")
    for kd in ("random", "PDB", "learned"):
        s = [r for r in rows if r["kind"] == kd]
        if not s:
            continue
        mm = np.array([r["measured"] for r in s])
        pp = np.array([r["predicted"] for r in s])
        print(f"      {kd:<8} n={len(s):>3}  r={pearsonr(mm, pp).statistic:+.3f}  "
              f"mean|err|={np.abs(mm - pp).mean():.4f}")
    print(f"      {'all':<8} n={len(rows):>3}  r={pearsonr(m, p).statistic:+.3f}  "
          f"mean|err|={np.abs(m - p).mean():.4f}")


def main():
    base = baseline()
    out = {"baseline_rev": BASELINE_REV}
    print("=" * 74)
    print("TICKET 01: what the corpus defects cost the published numbers")
    print("=" * 74)

    print("\n--- TIE INFLATION: the figure quoted in the abstract ---")
    b = [r["inflation"] for r in base["tie_inflation"]]
    n = [r["inflation"] for r in tie_inflation()]
    print(f"  committed baseline  n={len(b)}  mean {np.mean(b):+.4f}  max {max(b):+.4f}")
    print(f"  recomputed now      n={len(n)}  mean {np.mean(n):+.4f}  max {max(n):+.4f}")
    delta = float(np.mean(n) - np.mean(b))
    print(f"  difference in mean  {delta:+.{MOVED_DP}f}")
    print("  Reads no corpus and covers cube configurations only, so neither the")
    print("  deduplication defect nor the domain-pooling defect can reach it.")
    out["tie_inflation"] = {"baseline_mean": float(np.mean(b)), "now_mean": float(np.mean(n)),
                            "baseline_max": float(max(b)), "now_max": float(max(n)),
                            "delta": delta,
                            # Same precision the difference is printed at, so the
                            # console and the JSON cannot disagree about whether
                            # a published number moved.
                            "moved": bool(round(delta, MOVED_DP) != 0.0)}

    print("\n--- PREDICTING POOLED TAU: what the missing deduplication costs ---")
    raw, ded = corpus(False), corpus(True)
    report_prediction(raw, "as committed, no dedup:")
    report_prediction(ded, "with dedup applied:    ")
    out["prediction"] = {
        "n_raw": len(raw), "n_dedup": len(ded),
        "by_kind_raw": dict(Counter(r["kind"] for r in raw)),
        "by_kind_dedup": dict(Counter(r["kind"] for r in ded)),
    }

    print("\n--- SPREAD GROWTH: what the missing domain split costs ---")
    mom = corpus(True, min_shells=4, want_moments=True)
    print("  committed baseline, from before the sliding tile existed:")
    for k, v in sorted(base["beta_by_class"].items()):
        print(f"      {k:<8} n={v['n']:>3}  beta {v['beta_median']:.3f}")
    print("  pooled across domains, as the committed code now does:")
    for kd in ("PDB", "learned"):
        s = [r["beta"] for r in mom if r["kind"] == kd]
        if s:
            print(f"      {kd:<8} n={len(s):>3}  beta {np.median(s):.3f}")
    print("  split by domain, the correction:")
    split = {}
    for dom in ("cube", "tile"):
        for kd in ("PDB", "learned"):
            s = [r["beta"] for r in mom if r["kind"] == kd and r["domain"] == dom]
            if s:
                split[f"{dom}/{kd}"] = {"n": len(s), "beta": float(np.median(s))}
                print(f"      {dom:<5} {kd:<8} n={len(s):>3}  beta {np.median(s):.3f}")
    out["beta"] = {"baseline": base["beta_by_class"], "split": split}

    (RESULTS / "defect-cost.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n  -> defect-cost.json")


if __name__ == "__main__":
    sys.exit(main())
