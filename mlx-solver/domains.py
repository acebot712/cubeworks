"""One interface over both state spaces, so the measurements share a code path.

The scripts that produce the paper's numbers were written when there was only
the cube, and they reached for cube-specific machinery directly: `Task`,
`exact.indexer`, the `exact_k{k}.npy` naming, and a `project` that knows
DONT_CARE is 24. None of that is true of the sliding tile.

The tempting fix is a tile-flavoured copy of each script. That would be wrong.
The whole point of a second domain is to ask whether the SAME measurement gives
the same answer somewhere else, and two scripts that merely look alike do not
establish that: any difference in shell binning, in pair sampling, or in how
ties are counted would show up as a domain effect. So the scripts stay single
copies, and a measurement script takes whichever task it is handed.

WHERE THE BEHAVIOUR LIVES. On the tasks. Both kinds answer the same questions
about themselves, and a caller asks the task rather than a function here:

    task.domain          a short label, so a mixed set of runs stays sortable
    task.cells           the index space, and so a closed list's length
    task.min_rung        the lowest rung with a proper abstraction below it
    task.rank(states)    a unique index per state, into that task's table
    task.project(s, j)   the rung-j view of these states
    task.abstract(j)     rung j as a task in its own right
    task.table_path()    where its exact distance table lives
    task.histogram_path()  and where that table's BFS distance histogram lives
    task.rebuild_hint()  the command that builds both

There used to be a forwarding function here for each of those. They existed so
that call sites could migrate one at a time; every caller asks the task now, so
they are gone. Nothing dispatches on domain any more, in this module or any
other, which is what makes the no-branching rule easy to keep rather than a
thing to remember.

What is left is what belongs to neither task:

    make_task(name)      the naming rule, and the only place either class is
                         named. Anything starting with `tile-` is a sliding-tile
                         board, everything else is a cube sub-problem.
    rungs(task)          which abstractions are usable, including the
                         degenerate-rung test, which is the same question in
                         both domains
    missing_table(task)  why a task has no table, for a caller that can carry on
    load_table(task)     the file read, or an exit naming what is missing
"""
import numpy as np

from davi import Task
from tile import TileTask


def domain_of_name(name):
    """Which state space a task NAME belongs to, without building the task.

    `task.domain` is the answer when you hold a task. This is for the callers
    that hold only a name: a result file being read back, a row keyed by task.
    Building a task to read a label would mean loading a permutation table off
    disk for a string comparison.

    It is the same rule `make_task` dispatches on, deliberately, and they are
    adjacent so they cannot drift. Five scripts had their own copy of this line,
    each written as `str(task).startswith("tile-")`, which answers "cube" for
    every task OBJECT because neither class defines __str__. All five happened
    to be handed names, so none of them was wrong. With one copy left the trap
    is worth closing rather than repeating.
    """
    if not isinstance(name, str):
        raise TypeError(f"domain_of_name takes a task NAME, got {type(name).__name__}; "
                        f"a task knows its own domain, so ask it for .domain")
    return "tile" if name.startswith("tile-") else "cube"


def make_task(name, moves="all"):
    if domain_of_name(name) == "tile":
        return TileTask(name, moves)
    return Task(name, moves)


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
        sub = task.abstract(j)
        if not sub.table_path().exists():
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


def missing_table(task):
    """Why this task has no exact distance table, or None when it has one.

    Returned rather than raised because not every caller has to stop. An
    evaluation still has a solve rate and a saturation curve without ground
    truth; only the optimality block needs it. What a caller must not do is drop
    that block in silence, and one holding the sentence can say why instead.

    Two reasons, one shape: the task carries no index at all, or the file for its
    index is absent. `table_path` raises the first as a ValueError, which is the
    only thing it raises, so catching it here turns both into a message.
    """
    try:
        p = task.table_path()
    except ValueError as exc:
        return str(exc)
    if p.exists():
        return None
    return f"no {p.name}; build it with {task.rebuild_hint()}"


def load_table(task):
    """The exact distance table, or exit naming what is missing and what builds it."""
    why = missing_table(task)
    if why:
        raise SystemExit(why)
    return np.load(task.table_path())


# ------------------------------------------------------------------- selftest
# This module's docstring calls itself the thing the two-domain argument rests
# on, and it had no check at all until now.
INTERFACE = ("domain", "cells", "min_rung", "rank", "project", "abstract",
             "table_path", "histogram_path", "rebuild_hint")


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

    # THE COMPOSITION EVERY PATTERN-DATABASE CALL SITE WRITES. Since the
    # forwarders went, each caller spells this out itself:
    #
    #     sub = task.abstract(j); sub.rank(task.project(states, j))
    #
    # It has to work identically in both domains, because it is the one thing
    # the shared measurement scripts do that touches a domain difference. The
    # projected state must be a state OF the abstraction, so its index has to
    # land inside the abstraction's own table rather than the full task's.
    for name, mv in pairs:
        t = make_task(name, moves=mv)
        rng = np.random.default_rng(0)
        st, _ = t.scramble(256, 12, rng)
        j = rungs(t)[0]
        sub = t.abstract(j)
        idx = sub.rank(t.project(st, j))
        good = (idx.min() >= 0 and idx.max() < sub.cells
                and sub.cells <= t.cells
                and make_task(sub.name, moves=mv).name == sub.name
                # An abstraction forgets, so it cannot separate more states than
                # the task it abstracts: distinct projections are never more
                # numerous than the distinct states they came from.
                and len(np.unique(idx)) <= len(np.unique(t.rank(st))))
        print(f"  {name:<10} a projection indexes into its own abstraction   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

    # rank is an index, so it must be injective and inside the table it addresses
    for name, mv in pairs:
        t = make_task(name, moves=mv)
        rng = np.random.default_rng(1)
        st, _ = t.scramble(2000, 30, rng)
        idx = t.rank(st)
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

    # THE NAMESPACE COLLISION. A board's rung number lives in the cube's rung
    # namespace, so a caller composing the cube's name for tile-3x3 asks for
    # exact_k8.npy: not a missing file but a DIFFERENT task's table. Every table
    # a task names must therefore belong to that task.
    named_right = True
    for name in ("wings-k4", "wings-k6", "tile-3x3", "tile-2x4", "tile-2x3"):
        t = make_task(name)
        stem = t.table_path().stem
        want = name.startswith("tile-")
        named_right &= (stem.startswith("exact_tile-") == want)
    print(f"  a table name belongs to the domain that asked for it   "
          f"{'OK' if named_right else '*** FAIL ***'}")
    ok &= named_right

    # The histogram is written by one script per domain and read by another, so
    # the two agree only if both go through the task. Every histogram a task
    # names must be a file that is actually there.
    on_disk = all(make_task(n).histogram_path().exists()
                  for n in ("wings-k4", "wings-k6", "tile-3x3", "tile-2x4"))
    print(f"  a task names a histogram the writer actually wrote   "
          f"{'OK' if on_disk else '*** FAIL ***'}")
    ok &= on_disk

    # missing_table answers where load_table exits, and stays silent where the
    # table is there. A caller that can carry on needs the sentence, not a raise.
    have = make_task("tile-3x3")
    gone = make_task("wings-k20")
    none_at_all = make_task("centers")
    reasons = (missing_table(have) is None
               and gone.table_path().name in (missing_table(gone) or "")
               and "no numbered pieces" in (missing_table(none_at_all) or ""))
    print(f"  missing_table names the reason and is silent when there is none   "
          f"{'OK' if reasons else '*** FAIL ***'}")
    ok &= reasons

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
