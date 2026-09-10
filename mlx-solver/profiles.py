"""Does a learned heuristic degrade with distance in a way a classical one does not?

This is the paper's mechanism experiment, and it is a direct comparison against
the closest prior measure.

Wilt & Ruml (JAIR 57, 2016) introduced Goal Distance Rank Correlation: Kendall's
tau between a heuristic and exact distance-to-goal, computed offline, shown to
predict search cost. It is a single pooled scalar over all states. That is the
right object for the heuristics they studied: pattern databases, whose quality
has no particular reason to vary with distance from the goal.

A heuristic trained by bootstrapping is different by construction. Value is
anchored only at the goal and propagates outward, so accuracy should decay with
distance. A pooled tau averages that decay away: two heuristics with the same
GDRC can have completely different profiles, one uniform and one collapsing
exactly where search operates.

So this script computes, on the same rung and against the same exact ground truth:

  learned    the DAVI-trained network
  PDB        a genuine pattern database, the exact distance table of a SMALLER
             rung, which is admissible on the larger one because tracking fewer
             pieces can only shorten the required solution
  random     a control, to show what a flat-at-chance profile looks like

and reports each one's pooled GDRC alongside its per-shell ranking profile. The
claim is falsified if the PDB's profile decays like the learned one's, or if the
learned profile is flat.

    ../.venv-mlx/bin/python profiles.py --task wings-k6 --tag _s0 --pdb-k 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from domains import load_table, rungs
from evaluate import RESULTS, j_of, load
from resolution import sample_states

HERE = Path(__file__).parent


def sweep_sample(task, per_len, max_len, rng):
    """States drawn across a range of scramble lengths, not from deep walks.

    Deep random walks approximate the uniform distribution over the rung, which
    sounds right but concentrates almost every state into the two or three
    shells nearest the mean distance, so the shallow half of the profile is
    measured on a handful of states or not at all. Sweeping the scramble length
    populates the whole range.

    The scramble length is NOT the label. True distance is looked up in the
    exact table afterwards: a state scrambled 9 moves is usually nearer than 9,
    and treating the two as the same is the error this measurement exists to
    avoid. This matches profile_probed.py so profiles from the two are
    comparable.
    """
    out = []
    for L in range(1, max_len + 1):
        st = np.tile(task.solved, (per_len, 1))
        for _ in range(L):
            st = task.apply(st, rng.integers(0, task.n_moves, size=per_len))
        out.append(st)
    return np.concatenate(out)


def redraw(task, sampler, sampler_args):
    """The sample a profile recorded, drawn again. -> states, or None if unknown.

    The one place that turns a recorded sampler back into states, so an analysis
    reconstructing a profile's sample cannot drift from the code that drew it.
    None rather than a raise for an unrecognised sampler: a caller reconstructing
    an old profile has somewhere to go, and one drawing a new sample never gets
    here.
    """
    a = sampler_args or {}
    # The seed is required, not defaulted. default_rng(None) seeds from the OS
    # and would hand back a different sample every call, which is the one thing
    # a redraw must never do.
    if sampler == "sweep" and {"per_len", "max_len", "seed"} <= set(a):
        return sweep_sample(task, a["per_len"], a["max_len"],
                            np.random.default_rng(a["seed"]))
    if sampler == "walk" and {"n", "walk", "seed"} <= set(a):
        return sample_states(task, a["n"], a["walk"], np.random.default_rng(a["seed"]))
    return None


def _sizes(rows):
    """Shell sizes from one heuristic's profile rows: each pair's two endpoints."""
    if not rows or "n_lo" not in rows[0]:
        return None
    sizes = {r["d"]: r["n_lo"] for r in rows}
    sizes[rows[-1]["d"] + 1] = rows[-1]["n_hi"]
    return sizes


def recorded_shell_sizes(record):
    """The per-shell counts a profile recorded, as {d: count}, or None.

    Takes either shape: a Profile as written to disk, which holds every
    heuristic under `heuristics`, or one corpus record, which is a single
    heuristic flattened out of it. Both carry the same counts, because every
    heuristic in a Profile was measured on the same states.

    A Profile stores those counts twice over: per heuristic as the endpoints of
    each adjacent-shell pair, and as the list of shells that cleared
    --min-shell. The first is exact and is what a redraw is checked against; the
    second is all the oldest Profiles carry.
    """
    flat = _sizes(record.get("profile"))
    if flat:
        return flat
    for h in record.get("heuristics", {}).values():
        sizes = _sizes(h.get("profile"))
        if sizes:
            return sizes
    return None


def profile(h, true_d, shells, rng, pairs=20000):
    """Per-shell ranking accuracy: P[h(s) < h(s')] for s at d and s' at d+1.

    Alongside the accuracy we record the two moments of each shell. They are not
    decoration: `acc` turns out to be predicted, with no free parameters, by
    Phi(gap / (sd * sqrt(2))) -- the probability that one draw beats another
    under equal-variance normals. See dprime_law.py. So the pair (between-shell
    gap, within-shell spread) is the compressed form of the whole profile, and
    it is what an intervention has to move.
    """
    rows = []
    for d in shells[:-1]:
        sel_lo, sel_hi = true_d == d, true_d == d + 1
        a, b = h[sel_lo], h[sel_hi]
        if len(a) < 50 or len(b) < 50:
            continue
        m = min(pairs, len(a) * len(b))
        ia, ib = rng.integers(0, len(a), size=m), rng.integers(0, len(b), size=m)
        acc = float(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
        row = {"d": d, "acc": acc,
               "mean_lo": float(a.mean()), "mean_hi": float(b.mean()),
               "sd_lo": float(a.std()), "sd_hi": float(b.std()),
               "n_lo": int(a.size), "n_hi": int(b.size)}

        # SPLIT-SAMPLE VERSION. `acc` above and the moments beside it are computed
        # from the same states, and that shared sample manufactures agreement: run
        # the estimator on a heuristic with no signal at all and it reports a
        # correlation near +0.95 where the truth is zero, because a shell whose
        # sampled mean happens to sit high produces both a high predicted accuracy
        # and a high measured one. Splitting each shell in half -- moments from A,
        # accuracy from B -- removes that path entirely. The split figures are the
        # ones any claim about the two-moment law should be based on.
        half = np.zeros(true_d.shape[0], dtype=bool)
        half[::2] = True
        aA, bA = h[sel_lo & half], h[sel_hi & half]
        aB, bB = h[sel_lo & ~half], h[sel_hi & ~half]
        if min(aA.size, bA.size, aB.size, bB.size) >= 25 and aA.std() + bA.std() > 0:
            mm = min(pairs, aB.size * bB.size)
            ja, jb = rng.integers(0, aB.size, size=mm), rng.integers(0, bB.size, size=mm)
            row |= {"acc_split": float(np.mean(aB[ja] < bB[jb])
                                       + 0.5 * np.mean(aB[ja] == bB[jb])),
                    "mean_lo_a": float(aA.mean()), "mean_hi_a": float(bA.mean()),
                    "sd_lo_a": float(aA.std()), "sd_hi_a": float(bA.std())}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pdb-k", default="", help="comma list of abstraction rungs; "
                    "default = every available one. Reporting all of them is the point: "
                    "a single weak abstraction looks flat merely because it is near chance, "
                    "so the comparison must be made at matched overall strength.")
    ap.add_argument("--sampler", choices=["sweep", "walk"], default="sweep",
                    help="sweep = states across scramble lengths 1..max-len, which "
                         "populates every shell and matches profile_probed.py; "
                         "walk = deep random walks, which concentrate near the mean")
    ap.add_argument("--per-len", type=int, default=2500, help="states per scramble length")
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--n", type=int, default=40000, help="walk sampler only")
    ap.add_argument("--walk", type=int, default=200, help="walk sampler only")
    ap.add_argument("--min-shell", type=int, default=50, help="skip shells thinner than this")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    exact = load_table(task)

    pdb_ks = rungs(task, args.pdb_k)
    if not pdb_ks:
        raise SystemExit("no abstraction tables available for a PDB baseline")

    rng = np.random.default_rng(args.seed)

    # PROVENANCE. Recording only the sampler's name and the state count is not
    # enough to redraw the sample: n is a product, and 112500 is 2500x45 or
    # 4500x25, which are different distributions over shells. An analysis that
    # needs the full distance histogram has to redraw, so the arguments travel
    # with the profile.
    sampler_args = ({"per_len": args.per_len, "max_len": args.max_len, "seed": args.seed}
                    if args.sampler == "sweep"
                    else {"n": args.n, "walk": args.walk, "seed": args.seed})
    states = (sweep_sample(task, args.per_len, args.max_len, rng)
              if args.sampler == "sweep"
              else sample_states(task, args.n, args.walk, rng))
    true_d = exact[task.rank(states)].astype(np.int32)

    hs = {"learned": j_of(task, net, states).astype(np.float64)}
    for j in pdb_ks:
        sub = task.abstract(j)
        tab = load_table(sub)
        hs[f"PDB(k={j})"] = tab[sub.rank(task.project(states, j))].astype(np.float64)
    hs["random"] = rng.random(states.shape[0])

    uniq, cnt = np.unique(true_d, return_counts=True)
    shells = [int(d) for d, c in zip(uniq, cnt) if d > 0 and c >= args.min_shell]
    res = {"task": args.task, "tag": args.tag, "k": task.k, "moves": task.moveset,
           "domain": task.domain,
           "states": task.size, "step": meta.get("step"), "pdb_k": pdb_ks,
           "sampler": args.sampler, "sampler_args": sampler_args,
           "n": int(states.shape[0]),
           "shells": shells, "heuristics": {}}

    print(f"{args.task}{args.tag}  k={task.k} ({task.moveset})  {task.size:.2e} states  "
          f"step {meta.get('step'):,}")
    print(f"  {states.shape[0]:,} states ({args.sampler} sampler), exact ground truth, "
          f"PDB abstractions: {pdb_ks}")
    print(f"  shells with >= {args.min_shell} states: {shells}\n")

    for name, h in hs.items():
        # GDRC as Wilt & Ruml define it: one pooled Kendall tau over everything
        tau = float(kendalltau(h, true_d).statistic)
        rows = profile(h, true_d, shells, rng)
        # Span is recorded as context, NOT as a health check: a heuristic with a
        # small span and clean ordering is fine. What breaks a heuristic is
        # within-shell spread exceeding the between-shell gap, and the only thing
        # that measures is the adjacent-shell accuracy in `rows` below.
        span = float(np.ptp(h))
        res["heuristics"][name] = {"gdrc": tau, "span": span, "profile": rows}
        accs = [r["acc"] for r in rows]
        drop = (accs[0] - accs[-1]) if len(accs) > 1 else 0.0
        print(f"  {name:<12} GDRC {tau:+.4f}  span {span:8.3f}  "
              f"per-shell acc {accs[0]:.3f} -> {accs[-1]:.3f}  (drop {drop:+.3f})"
              + (f"   mean {np.mean(accs):.3f}"))

    print(f"\n  {'d':>3}  " + "".join(f"{n:>14}" for n in hs))
    for i, d in enumerate(shells[:-1]):
        line = f"  {d:>3}  "
        for n in hs:
            rows = res["heuristics"][n]["profile"]
            m = next((r for r in rows if r["d"] == d), None)
            line += f"{m['acc']:>14.3f}" if m else f"{'-':>14}"
        print(line)

    out = Path(args.out) if args.out else RESULTS / f"profile-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  -> {out.name}")


# ------------------------------------------------------- sampler provenance
# Profiles written before `sampler_args` existed record only the sampler's name
# and n. The arguments are not lost, though: a candidate reconstruction can be
# CHECKED against the shell sizes the profile itself recorded, and a wrong pair
# would have to reproduce twenty independent counts by coincidence. So this
# recovers them with proof rather than guessing, and refuses to write anything
# it cannot reproduce.
LEGACY_PER_LEN, LEGACY_SEED, LEGACY_MIN_SHELL = 2500, 7, 50


def check_sample(task, record, sampler_args):
    """Does redrawing with these arguments reproduce the profile's own shells?

    -> (True, how) or (False, why). This is what makes a recovered argument
    trustworthy and a recorded one worth trusting: the profile stores the shell
    sizes it measured, and a redraw either lands on them exactly or does not.
    """
    states = redraw(task, record.get("sampler"), sampler_args)
    if states is None:
        return False, f"no redraw for sampler {record.get('sampler')!r} with {sampler_args}"
    if states.shape[0] != record.get("n"):
        return False, (f"redraw gives {states.shape[0]} states, the profile "
                       f"recorded {record.get('n')}")
    hist = np.bincount(load_table(task)[task.rank(states)].astype(np.int64))
    sizes = recorded_shell_sizes(record)
    if sizes:
        wrong = {d: (n, int(hist[d]) if d < len(hist) else 0)
                 for d, n in sizes.items() if d >= len(hist) or hist[d] != n}
        if wrong:
            return False, f"{len(wrong)} of {len(sizes)} shell sizes disagree: {dict(list(wrong.items())[:3])}"
        return True, f"{len(sizes)} shell sizes"
    # The three oldest profiles predate the per-shell moments and carry only the
    # list of shells that cleared --min-shell. Weaker, still a real check.
    got = [d for d in range(1, len(hist)) if hist[d] >= LEGACY_MIN_SHELL]
    if got != record.get("shells"):
        return False, f"shell list disagrees: {got} against {record.get('shells')}"
    return True, f"the shell list ({len(got)} entries)"


def record_sampler_args(write):
    """Verify every profile's recorded sampler, and recover it where it is absent.

    Idempotent: a profile that already carries `sampler_args` is re-verified
    rather than rewritten. Run it after adding profiles to confirm each one
    still describes the sample it was measured on.
    """
    from domains import make_task, missing_table

    changed = failed = verified = 0
    for f in sorted(RESULTS.glob("profile-*.json")):
        d = json.loads(f.read_text())
        task = make_task(d["task"], moves=d["moves"])
        why = missing_table(task)
        if why:
            print(f"  {f.stem:<44} SKIP, {why}")
            continue
        have = d.get("sampler_args")
        # The candidate for a legacy sweep: n is per-len times max-len, and every
        # profile in this repository used the default per-len.
        cand = have or ({"per_len": LEGACY_PER_LEN,
                         "max_len": d["n"] // LEGACY_PER_LEN, "seed": LEGACY_SEED}
                        if d.get("sampler") == "sweep" and d.get("n", 0) % LEGACY_PER_LEN == 0
                        else None)
        good, how = check_sample(task, d, cand) if cand else (False, "no candidate arguments")
        if not good:
            failed += 1
            print(f"  {f.stem:<44} *** {how}")
            continue
        if have:
            verified += 1
            print(f"  {f.stem:<44} verified against {how}")
            continue
        print(f"  {f.stem:<44} recovered {cand}, checked against {how}")
        changed += 1
        if write:
            # Rebuilt key by key so the recovered arguments sit beside `sampler`
            # rather than at the end, matching what profiles.py writes today.
            out = {}
            for k, v in d.items():
                out[k] = v
                if k == "sampler":
                    out["sampler_args"] = cand
            f.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n  {verified} already recorded and re-verified, {changed} recovered"
          f"{'' if write else ' (dry run, nothing written)'}, {failed} unreproducible")
    return 1 if failed else 0


# ------------------------------------------------------------------- selftest
def selftest():
    """Checks on the redraw, which everything above rests on.

    The claim being tested is that `check_sample` DISCRIMINATES. If it accepted
    a wrong construction, recovering a legacy profile's arguments would be
    guessing with a rubber stamp, and a redrawn histogram could belong to a
    different sample than the one a profile was measured on.
    """
    from domains import load_table, make_task

    ok = True
    task = make_task("tile-2x3")
    truth, right = load_table(task), {"per_len": 300, "max_len": 12, "seed": 7}

    # A profile of a known sample, built here rather than read, so the check has
    # something with an answer we already know.
    states = redraw(task, "sweep", right)
    hist = np.bincount(truth[task.rank(states)].astype(np.int64))
    shells = [int(d) for d in range(1, len(hist)) if hist[d] >= 50]
    record = {"sampler": "sweep", "sampler_args": right, "n": int(states.shape[0]),
              "shells": shells,
              "heuristics": {"h": {"profile": [
                  {"d": d, "n_lo": int(hist[d]), "n_hi": int(hist[d + 1])}
                  for d in shells[:-1]]}}}

    good, how = check_sample(task, record, right)
    print(f"  the arguments that drew the sample are accepted   "
          f"{'OK' if good else '*** FAIL, ' + how + ' ***'}")
    ok &= good

    # Same n, different shape. This is the ambiguity the whole thing exists for:
    # 3600 is 300x12 and also 900x4, and they are different distributions.
    wrong = [({"per_len": 900, "max_len": 4, "seed": 7}, "same n, different shape"),
             ({"per_len": 300, "max_len": 12, "seed": 8}, "right shape, wrong seed"),
             ({"per_len": 300, "max_len": 11, "seed": 7}, "one scramble length short"),
             ({"per_len": 300, "max_len": 12}, "no seed recorded")]
    for args, label in wrong:
        rejected = not check_sample(task, record, args)[0]
        print(f"  rejects {label:<28} {'OK' if rejected else '*** FAIL, accepted ***'}")
        ok &= rejected

    # The two record shapes carry the same counts, so the check must not care
    # which it is handed: a corpus record is one heuristic flattened out.
    flat = {**record, "profile": record["heuristics"]["h"]["profile"]}
    del flat["heuristics"], flat["shells"]
    same = (recorded_shell_sizes(flat) == recorded_shell_sizes(record)
            and check_sample(task, flat, right)[0])
    print(f"  a flattened corpus record checks the same as the file   "
          f"{'OK' if same else '*** FAIL ***'}")
    ok &= same

    # The walk sampler has no profile on disk exercising it, which makes it the
    # branch most likely to be silently wrong. It must redraw, and repeatably.
    wa = {"n": 400, "walk": 30, "seed": 5}
    w1, w2 = redraw(task, "walk", wa), redraw(task, "walk", wa)
    walks = (w1 is not None and w1.shape == (400, task.n_slots)
             and np.array_equal(w1, w2))
    print(f"  the walk sampler redraws, and twice the same   "
          f"{'OK' if walks else '*** FAIL ***'}")
    ok &= walks

    # An unknown sampler has no redraw, and must say so rather than return
    # something plausible from the wrong branch.
    named = not check_sample(task, {**record, "sampler": "halton"}, right)[0]
    print(f"  an unknown sampler is refused, not approximated   "
          f"{'OK' if named else '*** FAIL ***'}")
    ok &= named

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--record-sampler-args", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    known, _ = ap.parse_known_args()
    if known.selftest:
        sys.exit(selftest())
    if known.record_sampler_args:
        sys.exit(record_sampler_args(known.write))
    main()
