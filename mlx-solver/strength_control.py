"""Does the decay belong to the training method, or just to heuristic strength?

An earlier version of this analysis compared the learned heuristic against a
single pattern database and concluded that only the learned one decays. That
conclusion was an artifact of which PDB was used. The abstraction was rung k=2,
whose pooled tau on rung k=6 is near chance -- and a heuristic near chance is
flat because it has nowhere to fall from, not because abstraction confers
distance-invariance. Reporting every available abstraction reverses the reading:
the strongest one is a PDB, and it decays more than the learned heuristic does.

So the confound is strength, and this script controls for it three ways:

  1. correlate decay against pooled GDRC across every observation, to establish
     that decay is a function of strength at all;
  2. regress decay on GDRC and compare the residuals of learned and PDB
     observations -- if the training method mattered, learned points would sit
     systematically above the line;
  3. match each learned heuristic to the abstraction nearest it in GDRC ON THE
     SAME STATES, and compare their decays pairwise.

Test 3 is the strict one: same task, same sampled states, same ground truth,
matched strength. Tests 1 and 2 pool across tasks and so mix depths.

    ../.venv-mlx/bin/python strength_control.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

import corpus

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"


def kind_of(name):
    return "learned" if name == "learned" else ("random" if name == "random" else "PDB")


def domain_of(task):
    return "tile" if str(task).startswith("tile-") else "cube"


def summarise(h):
    """Pooled strength, and how much ordering accuracy falls across the shells."""
    accs = [r["acc"] for r in h["profile"]]
    if len(accs) < 2:
        return None
    return {"gdrc": h["gdrc"], "decay": accs[0] - accs[-1], "mean_acc": float(np.mean(accs))}


def load():
    """Observations for the statistics, and the per-file grouping for the pairs.

    Two calls into the corpus rather than one, and the reason matters. The
    statistics need every heuristic counted once; the matched-strength pairs need
    every heuristic present on the file it was measured on, including the
    abstractions whose values repeat across files. A repeated abstraction value
    is still the right value for those states, so it belongs in a pair; it is
    only invalid to count it again as an independent observation.
    """
    def summarised(r):
        s = summarise({"gdrc": r["gdrc"], "profile": r["profile"]})
        if s is None:
            return None
        return s | {"file": r["file"], "task": r["task"], "tag": r["tag"],
                    "k": r["k"], "moves": r["moves"], "name": r["name"],
                    "kind": r["kind"], "domain": r["domain"], "final": r["final"]}

    recs, _, grouped = corpus.load_both_views("both")
    obs = [x for x in map(summarised, recs) if x is not None]
    per_file = {}
    for fname, rows in grouped.items():
        got = {n: summarised(r) for n, r in rows.items()}
        per_file[fname] = {n: v for n, v in got.items() if v is not None}
    return obs, per_file


def analyse(obs, per_file, label):
    """The three tests, inside ONE domain. Never across.

    Decay is first-minus-last MEASURED shell (docs/adr/0001), so a task profiled
    over 20 shells has more room to fall than one profiled over 4. Pooling the
    two compares measurement ranges rather than heuristics: the sliding tile
    averages 0.390 over a median 20 shells against the cube's 0.155 over 7, and
    pooling them once inverted this file's own null from Welch p = 0.478 to
    p = 0.042 with no new evidence behind it.
    """
    g = np.array([o["gdrc"] for o in obs])
    y = np.array([o["decay"] for o in obs])
    kinds = np.array([o["kind"] for o in obs])

    print(f"\n=== {label} ===")
    print(f"{len(obs)} observations from {len(per_file)} profiles "
          f"(k=2 excluded: no proper abstraction exists there)")

    by_kind = {}
    for kd in ("random", "PDB", "learned"):
        m = kinds == kd
        by_kind[kd] = {"n": int(m.sum()), "mean_gdrc": float(g[m].mean()),
                       "mean_decay": float(y[m].mean()),
                       "sd_decay": float(y[m].std(ddof=1))}
        print(f"  {kd:<8} n={m.sum():>3}  mean GDRC {g[m].mean():+.3f}  "
              f"mean decay {y[m].mean():+.3f} +- {y[m].std(ddof=1):.3f}")

    # 1. is decay a function of strength?
    r = float(np.corrcoef(g, y)[0, 1])
    n = len(g)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    from scipy.stats import t as tdist
    p_r = float(2 * tdist.sf(abs(t), n - 2))
    print(f"\n  decay vs strength: r = {r:+.3f}  (n={n}, p = {p_r:.2e})")

    # 2. residuals about that line, learned vs PDB
    slope, intercept = np.polyfit(g, y, 1)
    resid = y - (slope * g + intercept)
    rl, rp = resid[kinds == "learned"], resid[kinds == "PDB"]
    tt = ttest_ind(rl, rp, equal_var=False)
    print(f"  fit: decay = {slope:+.3f}*GDRC {intercept:+.3f}")
    print(f"  residual  learned {rl.mean():+.4f} +- {rl.std(ddof=1):.4f}   "
          f"PDB {rp.mean():+.4f} +- {rp.std(ddof=1):.4f}")
    # A null result is only worth reporting with its precision attached, so the
    # reader can see how large a real difference could still be hiding.
    se = np.sqrt(rl.var(ddof=1) / len(rl) + rp.var(ddof=1) / len(rp))
    diff = rl.mean() - rp.mean()
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    print(f"  Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}"
          f"  -> {'NO' if tt.pvalue > 0.05 else 'a'} detectable difference")
    print(f"  difference {diff:+.4f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] "
          f"against a mean decay of {y[kinds != 'random'].mean():.3f}")

    # 3. the strict test: matched on the same states. Restricted to final
    # checkpoints, so an intermediate snapshot of a run does not enter as a
    # second, near-identical pair alongside the run it came from.
    pairs = []
    for stem, rows in per_file.items():
        if "learned" not in rows or not rows["learned"]["final"]:
            continue
        pdbs = [v for k, v in rows.items() if v["kind"] == "PDB"]
        if not pdbs:
            continue
        L = rows["learned"]
        M = min(pdbs, key=lambda v: abs(v["gdrc"] - L["gdrc"]))
        pairs.append({"file": stem, "pdb": M["name"],
                      "gdrc_learned": L["gdrc"], "gdrc_pdb": M["gdrc"],
                      "gdrc_gap": abs(L["gdrc"] - M["gdrc"]),
                      "decay_learned": L["decay"], "decay_pdb": M["decay"]})
    d_l = np.array([p["decay_learned"] for p in pairs])
    d_p = np.array([p["decay_pdb"] for p in pairs])
    gaps = np.array([p["gdrc_gap"] for p in pairs])
    pdb_more = int((d_p > d_l).sum())
    print(f"\n  {len(pairs)} nearest-strength pairs on identical states "
          f"(median GDRC gap {np.median(gaps):.3f})")
    print(f"    PDB decays more in {pdb_more}/{len(pairs)}; "
          f"mean decay learned {d_l.mean():+.3f} vs PDB {d_p.mean():+.3f}")

    return {
        "domain": label,
        "n_obs": len(obs), "n_profiles": len(per_file), "by_kind": by_kind,
        "median_shells_note": "decay endpoints are the measured shells; "
                              "see docs/adr/0001",
        "decay_vs_strength_r": r, "decay_vs_strength_p": p_r,
        "fit": {"slope": float(slope), "intercept": float(intercept)},
        "residual_test": {"t": float(tt.statistic), "p": float(tt.pvalue),
                          "diff": float(diff), "ci95": [float(ci[0]), float(ci[1])],
                          "learned_resid": float(rl.mean()),
                          "learned_resid_sd": float(rl.std(ddof=1)),
                          "pdb_resid": float(rp.mean()),
                          "pdb_resid_sd": float(rp.std(ddof=1))},
        "matched_pairs": {"n": len(pairs), "pdb_decays_more": pdb_more,
                          "median_gdrc_gap": float(np.median(gaps)),
                          "mean_decay_learned": float(d_l.mean()),
                          "mean_decay_pdb": float(d_p.mean()),
                          "pairs": pairs},
    }


def main():
    obs, per_file = load()
    out = {"primary": "cube",
           "why_split": "Decay is not comparable across tasks with different "
                        "shell coverage; pooling inverted this control's null "
                        "result. See docs/adr/0001.",
           "by_domain": {}, "observations": obs}

    for dom in sorted({o["domain"] for o in obs}):
        sub = [o for o in obs if o["domain"] == dom]
        subf = {k: v for k, v in per_file.items()
                if v and next(iter(v.values()))["domain"] == dom}
        kinds = {o["kind"] for o in sub}
        if not {"learned", "PDB"} <= kinds:
            print(f"\n=== {dom} ===\n  skipped: needs both learned and PDB "
                  f"observations, has {sorted(kinds)}")
            continue
        out["by_domain"][dom] = analyse(sub, subf, dom)

    (RESULTS / "strength-control.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  -> strength-control.json  (domains: "
          f"{', '.join(out['by_domain']) or 'none'})")


if __name__ == "__main__":
    main()
