"""One interface over both state spaces, so the measurements share a code path.

The scripts that produce the paper's numbers were written when there was only
the cube, and they reach for cube-specific machinery directly: `Task`,
`exact.indexer`, the `exact_k{k}.npy` naming, and a `project` that knows
DONT_CARE is 24. None of that is true of the sliding tile.

The tempting fix is a tile-flavoured copy of each script. That would be wrong.
The whole point of a second domain is to ask whether the SAME measurement gives
the same answer somewhere else, and two scripts that merely look alike do not
establish that: any difference in shell binning, in pair sampling, or in how
ties are counted would show up as a domain effect. So the scripts stay single
copies and the domain differences are confined to the six functions here.

What a domain has to supply:

    make_task(name)        build it from a name
    table_path(task)       where its exact distance table lives
    rank(task, states)     a unique index per state, into that table
    rungs(task)            which abstractions have a table on disk
    project(task, s, j)    the rung-j view of these states
    abstract(task, j)      the task object for rung j, to rank the projection

Naming is the only thing a caller needs to know: anything starting with `tile-`
is a sliding-tile board, everything else is a cube sub-problem.

WHERE THE BEHAVIOUR LIVES. The tasks answer for themselves now: indexing,
projection onto a rung, index-space size, table location and how to rebuild it.
The functions below forward to them and exist so that every current caller keeps
working while the call sites migrate. What genuinely belongs here is what is not
a property of either task: the naming rule that constructs one, the file read,
and the degenerate-rung test, which is the same question in both domains.
"""
from pathlib import Path

import numpy as np

from davi import DONT_CARE as CUBE_DONT_CARE, Task
from exact import indexer
from tile import TileTask

HERE = Path(__file__).parent


def make_task(name, moves="all"):
    if name.startswith("tile-"):
        return TileTask(name, moves)
    return Task(name, moves)


def domain_of(task):
    """A short label for result files, so a mixed set of runs stays sortable."""
    return task.domain


def _cube_suffix(task):
    # The restricted generating sets change the distances, so they get their own
    # tables. "all" keeps the bare name for backwards compatibility with every
    # table already on disk.
    return "" if task.moveset == "all" else f"-{task.moveset}"


def table_path(task):
    return task.table_path()


def cells_of(task):
    """Size of the index space `rank` maps into, and so the closed list's length.

    Also the hard bound on expansions: a search with a closed list cannot expand
    a state twice, so it terminates within this many.
    """
    return task.cells


def rank(task, states):
    """Unique int64 index per state, matching the layout of table_path(task)."""
    return task.rank(states)


def abstract(task, j):
    """The task object for rung j of this domain."""
    return task.abstract(j)


def project(task, states, j):
    """Rung-j view: keep the first j tracked pieces, forget the rest.

    This is a pattern-database abstraction in both domains, so the resulting
    distance is an admissible lower bound on the true distance. The tile version
    also keeps the blank, which the cube has no analogue of; see TileTask.
    """
    return task.project(states, j)


def rungs(task, spec=""):
    """Abstraction rungs strictly weaker than `task`, whose tables exist.

    Reporting every available one is deliberate rather than thorough for its own
    sake: a single weak abstraction looks flat merely because it is near chance,
    so the learned-versus-classical comparison is only meaningful at matched
    overall strength, and that needs a ladder to match against.
    """
    if spec:
        return [int(x) for x in spec.split(",")]
    if not task.k:
        raise ValueError(f"{task.name} tracks no numbered pieces, so it has no "
                         f"rung ladder beneath it")
    lo = task.min_rung
    out = []
    for j in range(lo, task.k):
        sub = abstract(task, j)
        if not table_path(sub).exists():
            continue
        # DEGENERATE RUNG. An abstraction that can still tell every state apart
        # is not an abstraction, it is the oracle: its "estimate" is the exact
        # distance, so it scores a perfect 1.000 at every shell and a GDRC of
        # exactly 1. Handing that to the PDB comparison would not make the
        # classical baseline look strong, it would make it look omniscient, and
        # the matched-on-strength argument would be meaningless.
        #
        # It happens on the 8-puzzle at k=6: with only two tiles untracked, the
        # parity invariant fixes which way round they go, so knowing the other
        # seven positions determines the whole board. Reachable count is the
        # cheap way to see it, and the cube has the same structure at its top
        # rung, where the untracked pieces stop being able to absorb parity.
        if sub.size >= task.size:
            continue
        out.append(j)
    return out


def load_table(task):
    p = table_path(task)
    if not p.exists():
        raise SystemExit(f"need {p.name}; run {task.rebuild_hint()}")
    return np.load(p)


# ------------------------------------------------------------------- selftest
# This module's docstring calls itself the thing the two-domain argument rests
# on, and it had no check at all until now.
INTERFACE = ("domain", "cells", "min_rung", "rank", "project", "abstract",
             "table_path", "rebuild_hint")


def selftest():
    import numpy as np
    ok = True
    pairs = [("wings-k4", "all"), ("tile-3x3", "all")]

    for name, mv in pairs:
        t = make_task(name, moves=mv)
        missing = [a for a in INTERFACE if not hasattr(t, a)]
        print(f"  {name:<10} answers the whole interface   "
              f"{'OK' if not missing else '*** FAIL, missing ' + str(missing) + ' ***'}")
        ok &= not missing

    # The forwarders must agree with the tasks, or a half-migrated call site
    # would silently get a different answer from its neighbour.
    for name, mv in pairs:
        t = make_task(name, moves=mv)
        rng = np.random.default_rng(0)
        st, _ = t.scramble(64, 12, rng)
        j = rungs(t)[0]
        agree = (np.array_equal(rank(t, st), t.rank(st))
                 and np.array_equal(project(t, st, j), t.project(st, j))
                 and cells_of(t) == t.cells
                 and table_path(t) == t.table_path()
                 and abstract(t, j).name == t.abstract(j).name
                 and domain_of(t) == t.domain)
        print(f"  {name:<10} free functions forward to the same answers   "
              f"{'OK' if agree else '*** FAIL ***'}")
        ok &= agree

    # rank is an index, so it must be injective and inside the table it addresses
    for name, mv in pairs:
        t = make_task(name, moves=mv)
        rng = np.random.default_rng(1)
        st, _ = t.scramble(2000, 30, rng)
        idx = rank(t, st)
        uniq_states = len(np.unique(st, axis=0))
        good = len(np.unique(idx)) == uniq_states and idx.min() >= 0 and idx.max() < t.cells
        print(f"  {name:<10} rank is injective and within [0, cells)   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

    # A rung that still separates every state is the oracle, not a baseline. The
    # test is domain-neutral, which is why it stays here and not on the tasks.
    t = make_task("tile-3x3")
    kept = rungs(t)
    degenerate = [j for j in range(t.min_rung, t.k) if t.abstract(j).size >= t.size]
    good = kept and degenerate and not (set(kept) & set(degenerate))
    print(f"  degenerate rungs excluded: kept {kept}, dropped {degenerate}   "
          f"{'OK' if good else '*** FAIL ***'}")
    ok &= good

    # A missing table must name the command that builds it.
    t = make_task("tile-2x3")
    try:
        load_table(t.abstract(1))
        named = False
    except SystemExit as exc:
        named = "tile.py" in str(exc)
    print(f"  a missing table names its rebuild command   "
          f"{'OK' if named else '*** FAIL ***'}")
    ok &= named

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(selftest())
    raise SystemExit("give --selftest")
