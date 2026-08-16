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

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"


def kind_of(name):
    return "learned" if name == "learned" else ("random" if name == "random" else "PDB")


def summarise(h):
    """Pooled strength, and how much ordering accuracy falls across the shells."""
    accs = [r["acc"] for r in h["profile"]]
    if len(accs) < 2:
        return None
    return {"gdrc": h["gdrc"], "decay": accs[0] - accs[-1], "mean_acc": float(np.mean(accs))}


def load():
    """Every profile, plus the DEDUPLICATED observation set.

    A profile file is produced per learned checkpoint, and each one re-measures
    the same abstractions and the same random control on the same states. So a
    PDB appears once per checkpoint with an identical value every time. Counting
    those repeats as independent observations inflates n roughly fourfold and
    makes every p-value far too small. Identity is therefore (task, moveset,
    heuristic) for PDB and random -- which is all those depend on -- and
    additionally the tag for learned, whose weights differ per seed and step.
    """
    obs, per_file, seen = [], {}, {}
    for f in sorted(RESULTS.glob("profile-*.json")):
        # The loss-comparison runs (p3_losses.py) write profiles into the same
        # directory. They are a separate experiment with a different training
        # objective, and folding them in here would silently change the control
        # this file reports. dprime_law.py does include them, deliberately --
        # its claim is about all heuristics, not about this study's sample.
        if "_p3-" in f.stem:
            continue
        d = json.load(open(f))
        # A rung-k task has proper abstractions only for j < k. On k=2 the only
        # table available is the exact answer itself, which is not an abstraction
        # and would enter as a perfect heuristic; those files are excluded.
        if d["k"] <= 2:
            continue
        rows = {}
        for name, h in d["heuristics"].items():
            s = summarise(h)
            if s is None:
                continue
            kind = kind_of(name)
            s |= {"file": f.stem, "task": d["task"], "tag": d["tag"], "k": d["k"],
                  "moves": d["moves"], "name": name, "kind": kind,
                  "final": "@" not in d["tag"]}
            rows[name] = s
            key = (d["task"], d["moves"], name) + ((d["tag"],) if kind == "learned" else ())
            if key not in seen:
                seen[key] = s
                obs.append(s)
        per_file[f.stem] = rows
    return obs, per_file


def main():
    obs, per_file = load()
    g = np.array([o["gdrc"] for o in obs])
    y = np.array([o["decay"] for o in obs])
    kinds = np.array([o["kind"] for o in obs])

    print(f"{len(obs)} observations from {len(per_file)} profiles "
          f"(k=2 excluded: no proper abstraction exists there)\n")

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

    out = {
        "n_obs": len(obs), "n_profiles": len(per_file), "by_kind": by_kind,
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
        "observations": obs,
    }
    (RESULTS / "strength-control.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  -> strength-control.json")


if __name__ == "__main__":
    main()
