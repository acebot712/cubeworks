"""The Profile corpus: one place that decides what the analyses measure.

Four hand-written copies of this loop existed before this module, and they
disagreed. Two deduplicated and two did not. Two split by domain and two did
not. One applied the rung-size exclusion only to the cube, three applied it to
everything. Because those decisions lived in the copies, a fix reached whichever
copy its author happened to be editing, which is how the analyses ended up
counting pattern databases 73 times against 21 distinct heuristics and pooling
38 cube observations with 13 sliding-tile ones.

So the decisions live here instead, and none of them is optional:

  DEDUPLICATION is not a parameter. A Profile is written once per learned
  checkpoint and re-measures the same Rungs and the same random control on the
  same states, so those rows repeat verbatim. Identity is the task, the moveset
  and the heuristic name, plus the tag when the heuristic is learned, because
  learned weights differ per seed and per step while a Rung depends on nothing
  else.

  THE DOMAIN is a required argument, not a default. Pooling the cube with the
  sliding tile is sometimes right and usually wrong, and the caller has to say
  which it means. `both` is a word you have to type.

  THE RUNG-SIZE EXCLUSION knows that a sliding-tile task's rung counts tiles.
  A cube rung below three has no proper abstraction, so it is dropped; a board
  is not a cube rung and the same threshold means nothing there.

Records come back one per (Profile, heuristic), already deduplicated and tagged.

There is a second entry point, `load_by_file`, and it exists because one caller
genuinely needs the duplicates. The matched-strength pairs compare a learned
heuristic against the nearest abstraction measured on the SAME states, so both
members have to be present on the same file, and deduplication removes the
abstractions from every file after the first. That repeated value is still the
right value for those states; counting it again as an independent observation is
the only thing that was ever wrong. Two entry points rather than one flag,
because a flag would put the defect one default away.

    ../.venv-mlx/bin/python corpus.py --selftest
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"

DOMAINS = ("cube", "tile", "both")
EXPERIMENTS = ("main", "losses")
KINDS = ("learned", "PDB", "random")


def kind_of(name):
    """learned, random, or PDB. One definition, previously three."""
    if name == "learned":
        return "learned"
    return "random" if name == "random" else "PDB"


def domain_of(task):
    """Which state space. The same rule `domains.make_task` dispatches on."""
    return "tile" if str(task).startswith("tile-") else "cube"


def _excluded_rung(task, k):
    """A cube rung below 3 has no proper abstraction, so it cannot be a baseline.

    Domain-conditional on purpose. A sliding-tile task's `k` counts tiles, and
    applying a cube rung threshold to a board silently drops boards for a reason
    that does not apply to them. Three of the four original copies did exactly
    that; it was inert only because no small board was in the corpus yet.
    """
    return domain_of(task) == "cube" and k <= 2


def load(domain, experiment="main", results_dir=None):
    """Deduplicated, domain-tagged Profiles.

    `domain` is required and must be one of cube, tile, both. There is no
    default, because every default here is wrong for some caller and silence
    about it is what produced the pooling defect.

    Returns (records, stats). Each record is one heuristic measured on one
    Profile, carrying its file, task, moveset, tag, rung, domain, whether it is
    a Final checkpoint, its kind, its GDRC and its per-Shell array.
    """
    records, dropped, excluded = _scan(domain, experiment, results_dir, dedup=True)
    return records, _stats(domain, experiment, records, dropped, excluded)


def _stats(domain, experiment, records, dropped, excluded):
    return {"domain": domain, "experiment": experiment,
            "n_records": len(records), "n_files": len({r["file"] for r in records}),
            "dropped_duplicates": dropped, "excluded_files": excluded,
            "by_domain": {dm: sum(1 for r in records if r["domain"] == dm)
                          for dm in ("cube", "tile")},
            "by_kind": {kd: sum(1 for r in records if r["kind"] == kd)
                        for kd in KINDS}}


def _scan(domain, experiment, results_dir, dedup):
    """The one loop. Both entry points go through it."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")
    root = Path(results_dir) if results_dir else RESULTS
    want_losses = experiment == "losses"
    records, seen = [], set()
    dropped = excluded = 0

    for f in sorted(root.glob("profile-*.json")):
        # The loss-comparison runs are a separate experiment with a different
        # training objective. Which population you want is part of the request.
        if ("_p3-" in f.stem) != want_losses:
            continue
        d = json.load(open(f))
        task, moves, tag, k = d["task"], d["moves"], d["tag"], d["k"]
        dom = domain_of(task)
        if domain != "both" and dom != domain:
            continue
        if _excluded_rung(task, k):
            excluded += 1
            continue
        for name, h in d["heuristics"].items():
            kind = kind_of(name)
            key = (task, moves, name) + ((tag,) if kind == "learned" else ())
            if key in seen:
                dropped += 1
                if dedup:
                    continue
            seen.add(key)
            records.append({
                "file": f.stem, "task": task, "moves": moves, "tag": tag, "k": k,
                "domain": dom, "kind": kind, "name": name,
                # An Intermediate snapshot is named @step and is not independent
                # of the Final checkpoint of the run it came from.
                "final": "@" not in tag,
                "gdrc": h["gdrc"], "profile": h["profile"],
            })

    return records, dropped, excluded


def load_both_views(domain, experiment="main", results_dir=None):
    """Both views from ONE scan, for the caller that needs each.

    Returns (records, stats, by_file). Calling `load` and `load_by_file` in
    succession reads and parses every Profile twice, which is invisible at 29
    files and will not stay that way.
    """
    all_recs, dropped, excluded = _scan(domain, experiment, results_dir, dedup=False)
    seen, deduped = set(), []
    for r in all_recs:
        key = ((r["task"], r["moves"], r["name"])
               + ((r["tag"],) if r["kind"] == "learned" else ()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    grouped = {}
    for r in all_recs:
        grouped.setdefault(r["file"], {})[r["name"]] = r
    return deduped, _stats(domain, experiment, deduped, dropped, excluded), grouped


def load_by_file(domain, experiment="main", results_dir=None):
    """Every heuristic on every Profile, grouped by file and NOT deduplicated.

    For one purpose only: the matched-strength pairs, which pair a learned
    heuristic against the nearest abstraction MEASURED ON THE SAME STATES. That
    comparison needs both members present on the same file, and deduplication
    removes the abstractions from every file after the first, since their values
    repeat verbatim. The repeated value is still the right value for those
    states; it is only invalid to count it again as an independent observation.

    So this returns duplicates on purpose, and must never be used for a
    statistic. Use `load` for anything that counts. The separation is deliberate:
    a single function with a keep_duplicates flag would put the defect one
    default away.
    """
    out = {}
    records, _, _ = _scan(domain, experiment, results_dir, dedup=False)
    for r in records:
        out.setdefault(r["file"], {})[r["name"]] = r
    return out


# ------------------------------------------------------------------- selftest
def _write(dirpath, stem, task, moves, tag, k, heuristics):
    body = {"task": task, "moves": moves, "tag": tag, "k": k,
            "heuristics": {n: {"gdrc": g, "profile": [{"d": 1, "acc": 0.9}]}
                           for n, g in heuristics.items()}}
    (Path(dirpath) / f"{stem}.json").write_text(json.dumps(body))


def selftest():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # Two Final checkpoints of one cube task. Each re-measures the SAME two
        # abstractions and the SAME random control, so those repeat verbatim
        # while the learned rows are genuinely distinct.
        for tag in ("_s0", "_s1"):
            _write(tmp, f"profile-wings-k4{tag}", "wings-k4", "all", tag, 4,
                   {"learned": 0.85, "PDB(k=2)": 0.66, "PDB(k=3)": 0.81, "random": 0.00})
        # An Intermediate snapshot of the first run.
        _write(tmp, "profile-wings-k4_s0@3000", "wings-k4", "all", "_s0@3000", 4,
               {"learned": 0.79})
        # A sliding-tile board whose rung is 8. Three of the four original copies
        # would have kept it, but a fourth applied a cube threshold to it.
        _write(tmp, "profile-tile-3x3", "tile-3x3", "all", "", 8,
               {"learned": 0.96, "PDB(k=5)": 0.99, "random": 0.00})
        # A cube rung of 2, which has no proper abstraction and must be dropped.
        _write(tmp, "profile-wings-k2_s0", "wings-k2", "all", "_s0", 2,
               {"learned": 0.50, "random": 0.00})
        # A loss-comparison run, a separate experiment.
        _write(tmp, "profile-wings-k4_p3-l2_s0", "wings-k4", "all", "_p3-l2_s0", 4,
               {"learned": 0.88})

        rec, st = load("cube", results_dir=tmp)
        # 2 learned + 1 snapshot learned + 2 PDB + 1 random = 6
        want = {"learned": 3, "PDB": 2, "random": 1}
        good = st["by_kind"] == want and st["dropped_duplicates"] == 3
        print(f"  dedup: kinds {st['by_kind']} (want {want}), dropped "
              f"{st['dropped_duplicates']} (want 3)   {'OK' if good else '*** FAIL ***'}")
        ok &= good

        good = st["excluded_files"] == 1 and not any(r["task"] == "wings-k2" for r in rec)
        print(f"  cube rung 2 excluded: {st['excluded_files']} file   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

        tile, tst = load("tile", results_dir=tmp)
        kept = len(tile) == 3 and all(r["domain"] == "tile" for r in tile)
        print(f"  tile board with rung 8 NOT dropped by the cube threshold: "
              f"{len(tile)} records   {'OK' if kept else '*** FAIL ***'}")
        ok &= kept

        cube_only, _ = load("cube", results_dir=tmp)
        both, bst = load("both", results_dir=tmp)
        split = (all(r["domain"] == "cube" for r in cube_only)
                 and bst["by_domain"] == {"cube": len(cube_only), "tile": len(tile)})
        print(f"  domain selection: cube {len(cube_only)}, tile {len(tile)}, "
              f"both {len(both)}   {'OK' if split else '*** FAIL ***'}")
        ok &= split

        try:
            load(results_dir=tmp)
            defaulted = False
        except TypeError:
            defaulted = True
        try:
            load("everything", results_dir=tmp)
            rejected = False
        except ValueError:
            rejected = True
        print(f"  pooling cannot happen by default or by typo   "
              f"{'OK' if defaulted and rejected else '*** FAIL ***'}")
        ok &= defaulted and rejected

        finals = {r["tag"]: r["final"] for r in cube_only if r["kind"] == "learned"}
        marked = finals.get("_s0") and finals.get("_s1") and not finals.get("_s0@3000")
        print(f"  Final checkpoints separable from Intermediate snapshots   "
              f"{'OK' if marked else '*** FAIL ***'}")
        ok &= marked

        # Deduplicated, the second file keeps only its learned row. Grouped by
        # file it keeps all four, which is what the matched-strength pairs need:
        # both members of a pair measured on the same states.
        deduped_second = {r["name"] for r in cube_only
                          if r["file"] == "profile-wings-k4_s1"}
        grouped = load_by_file("cube", results_dir=tmp)
        first = grouped.get("profile-wings-k4_s0", {})
        second = grouped.get("profile-wings-k4_s1", {})
        good = (deduped_second == {"learned"} and len(first) == 4 and len(second) == 4)
        print(f"  deduplicated view drops the repeats ({sorted(deduped_second)}), "
              f"grouped view keeps them ({len(second)} on the second file)   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

        losses, lst = load("cube", experiment="losses", results_dir=tmp)
        good = len(losses) == 1 and losses[0]["tag"] == "_p3-l2_s0"
        print(f"  loss-comparison runs are a separate population   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", default="", help="cube | tile | both: summarise the real corpus")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.show:
        _, st = load(args.show)
        print(json.dumps(st, indent=2))
        return 0
    raise SystemExit("give --selftest or --show <domain>")


if __name__ == "__main__":
    sys.exit(main())
