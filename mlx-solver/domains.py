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


def is_tile(task):
    return isinstance(task, TileTask)


def domain_of(task):
    """A short label for result files, so a mixed set of runs stays sortable."""
    return "tile" if is_tile(task) else "cube"


def _cube_suffix(task):
    # The restricted generating sets change the distances, so they get their own
    # tables. "all" keeps the bare name for backwards compatibility with every
    # table already on disk.
    return "" if task.moveset == "all" else f"-{task.moveset}"


def table_path(task):
    if is_tile(task):
        return HERE / f"exact_{task.name}.npy"
    return HERE / f"exact_k{task.k}{_cube_suffix(task)}.npy"


def rank(task, states):
    """Unique int64 index per state, matching the layout of table_path(task)."""
    if is_tile(task):
        return task.rank(states)
    return indexer(task.k)(states)


def abstract(task, j):
    """The task object for rung j of this domain."""
    if is_tile(task):
        return TileTask(f"tile-{task.rows}x{task.cols}-k{j}")
    return Task(f"wings-k{j}", moves=task.moveset)


def project(task, states, j):
    """Rung-j view: keep the first j tracked pieces, forget the rest.

    This is a pattern-database abstraction in both domains, so the resulting
    distance is an admissible lower bound on the true distance. The tile version
    also keeps the blank, which the cube has no analogue of; see TileTask.
    """
    if is_tile(task):
        return task.project(states, j)
    out = states.copy()
    out[out >= j] = CUBE_DONT_CARE
    return out


def rungs(task, spec=""):
    """Abstraction rungs strictly weaker than `task`, whose tables exist.

    Reporting every available one is deliberate rather than thorough for its own
    sake: a single weak abstraction looks flat merely because it is near chance,
    so the learned-versus-classical comparison is only meaningful at matched
    overall strength, and that needs a ladder to match against.
    """
    if spec:
        return [int(x) for x in spec.split(",")]
    lo = 1 if is_tile(task) else 2
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
        how = (f"tile.py --board {task.rows}x{task.cols} --save-table" if is_tile(task)
               else f"exact.py --k {task.k} --moves {task.moveset} --save-table")
        raise SystemExit(f"need {p.name}; run {how}")
    return np.load(p)
