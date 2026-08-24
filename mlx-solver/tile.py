"""The sliding-tile puzzle, as a second domain for the heuristic measurements.

WHY THIS EXISTS. Every result in the paper is measured on Rubik's-cube
sub-problems. The two-moment law, the claim that pattern databases decay as much
as learned heuristics, the tau-b tie bias: all of it is one state space seen
through several sub-problems of itself. That is a single point of external
validity, and it is the first thing a referee should ask about. The sliding-tile
puzzle is the standard second domain in heuristic search, it is structurally
unlike the cube (a small branching factor, a locality the cube does not have,
and a move set that depends on the state rather than a fixed group action), and
exhaustive ground truth is available on several board sizes. If the law holds
here too, it is a fact about heuristics rather than a fact about the cube.

WHAT HAD TO CHANGE. Two things in the cube code do not carry over.

  1. `davi.Task` applies a move as a FIXED permutation of slots, `perms[m]`, the
     same one whatever the state is. That is what a group action means and it is
     true of the cube. It is not true here: sliding the blank up permutes a
     different pair of slots depending on where the blank is. So the move has to
     be computed from the state, which is what `apply` below does. Everything
     else about the interface is unchanged, so the downstream scripts do not
     care which domain they were handed.

  2. `exact.indexer` is mixed-radix base 24 over the tracked pieces' positions.
     It is branch-free and vectorised, which is why it was chosen, but it costs
     24^k cells for P(24,k) states and there is no equivalent slack to spend
     here: 12! is already 479 million cells and the wasteful version would not
     fit. `perm_rank` below is a true minimal ranking, [0, n!), and it is still
     branch-free: the Lehmer code needs, for each position, the number of later
     positions holding a smaller value, and that is an n-by-n comparison matrix
     masked to its upper triangle, not the sequential scan the exact.py comment
     assumed. n is at most 16, so the matrix is free.

BOARD SIZES. Reachable states are n!/2 on every board (exactly half of the
arrangements satisfy the parity invariant), so the ladder is set by n = rows *
cols and the table is n! bytes:

    2x2      4! =         24 cells,         12 reachable
    2x3      6! =        720 cells,        360 reachable
    2x4      8! =     40,320 cells,     20,160 reachable
    3x3      9! =    362,880 cells,    181,440 reachable
    3x4     12! = 479,001,600 cells, 239,500,800 reachable   (479 MB, slow)

3x3 is the workhorse: it is the 8-puzzle, its diameter is a published number
(31 moves), and that makes it a real check on this file rather than a
self-consistent one.

    ../.venv-mlx/bin/python tile.py --selftest
    ../.venv-mlx/bin/python tile.py --board 3x3 --save-table
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
UNSEEN = 255

# Slide the BLANK in this direction. Naming the mover as the blank rather than
# the tile avoids the usual ambiguity where "up" means opposite things in two
# papers. (d_row, d_col).
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def perm_rank(states):
    """Lehmer rank of each row, a bijection onto [0, n!).

    c[i] counts the positions after i holding a smaller value, and the rank is
    sum c[i] * (n-1-i)!. The count is an upper-triangular masked comparison
    matrix, so this is one vectorised expression with no Python loop over
    states and none over positions either.
    """
    B, n = states.shape
    st = states.astype(np.int16)
    later = np.triu(np.ones((n, n), dtype=bool), 1)          # j > i
    smaller = st[:, :, None] > st[:, None, :]                # state[i] > state[j]
    c = (smaller & later).sum(axis=2)                        # (B, n)
    fact = np.array([math.factorial(n - 1 - i) for i in range(n)], dtype=np.int64)
    return c @ fact


def perm_unrank(ranks, n):
    """Inverse of perm_rank. Only used by the self-test, so it may loop."""
    ranks = np.asarray(ranks, dtype=np.int64)
    out = np.empty((ranks.size, n), dtype=np.int8)
    for r, rank in enumerate(ranks):
        pool = list(range(n))
        rem = int(rank)
        for i in range(n):
            f = math.factorial(n - 1 - i)
            out[r, i] = pool.pop(rem // f)
            rem %= f
    return out


class TileTask:
    """A sliding-tile board, interface-compatible with davi.Task.

    STATE. One int8 per slot in row-major order, holding the tile that occupies
    that slot. Tile 0 is the blank. Solved is therefore `arange(n)`: blank in
    the top-left, tiles in reading order. Blank-last is the other common
    convention; blank-first is chosen only because it makes solved == arange(n)
    and so removes a source of off-by-one confusion when reading a state by eye.

    MOVES. Four, always, in a fixed order (U, D, L, R), sliding the blank. A
    move that would take the blank off the board leaves the state ALONE rather
    than being masked out. That self-loop is deliberate:

      - `children` stays a fixed (B, 4, n_slots) array, which is what every
        downstream caller already assumes;
      - BFS is unaffected, since a self-loop revisits a state already seen;
      - the Bellman backup is unaffected, because min over moves of 1 + J(s')
        can only be made larger by including 1 + J(s), never smaller, so a
        self-loop can never win the min against a real move that reduces J.

    The cost is that the branching factor reads as 4 when the effective one is
    between 2 and 4. Nothing here measures branching, so that is a fair trade,
    but any future search-cost experiment must not take n_moves at face value.
    """

    def __init__(self, name, moves="all"):
        if not name.startswith("tile-"):
            raise ValueError(f"not a tile task name: {name!r}")
        spec = name[len("tile-"):]
        try:
            rows, cols = (int(x) for x in spec.split("x"))
        except ValueError:
            raise ValueError(f"expected tile-RxC, got {name!r}") from None
        if rows < 2 or cols < 2:
            raise ValueError(f"board must be at least 2x2, got {rows}x{cols}")

        self.name = name
        self.rows, self.cols = rows, cols
        self.n_slots = rows * cols
        self.k = self.n_slots
        # One symbol per tile including the blank. Unlike the cube ladder there
        # is no DONT_CARE symbol yet: abstractions come later and will add one.
        self.n_sym = self.n_slots
        self.n_in = self.n_slots * self.n_sym
        self.solved = np.arange(self.n_slots, dtype=np.int8)

        self.moveset = moves
        if moves != "all":
            raise ValueError("the tile domain has one generating set; depth is "
                             "varied by board size, not by restricting moves")
        self.tokens = list(MOVES)
        self.n_moves = len(self.tokens)

        # Precompute, for every (blank slot, move), the slot the blank swaps
        # with, or -1 when the move runs off the board. This is the state
        # dependence the cube does not have, reduced to a lookup so that `apply`
        # is still branch-free.
        self.dest = np.full((self.n_slots, self.n_moves), -1, dtype=np.int64)
        for slot in range(self.n_slots):
            r, c = divmod(slot, cols)
            for m, tok in enumerate(self.tokens):
                dr, dc = MOVES[tok]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.dest[slot, m] = nr * cols + nc

        # Exactly half of the n! arrangements satisfy the parity invariant.
        self.size = math.factorial(self.n_slots) // 2
        self.cells = math.factorial(self.n_slots)

    # ---------------------------------------------------------------- dynamics
    def blank_of(self, states):
        return np.argmax(states == 0, axis=1)

    def apply(self, states, move_ids):
        out = states.copy()
        blank = self.blank_of(states)
        dest = self.dest[blank, move_ids]
        legal = dest >= 0
        if legal.any():
            rowi = np.flatnonzero(legal)
            b, d = blank[legal], dest[legal]
            out[rowi, b] = states[rowi, d]
            out[rowi, d] = 0
        return out

    def children(self, states):
        B = states.shape[0]
        out = np.empty((B, self.n_moves, self.n_slots), dtype=np.int8)
        for m in range(self.n_moves):
            out[:, m, :] = self.apply(states, np.full(B, m))
        return out

    def scramble(self, batch, k_max, rng):
        """Random legal walks. Every reachable state is solvable by construction,
        so unlike a shuffled board there is no parity check to make."""
        states = np.tile(self.solved, (batch, 1))
        ks = rng.integers(1, k_max + 1, size=batch)
        for step in range(k_max):
            active = ks > step
            if not active.any():
                break
            n = int(active.sum())
            states[active] = self.apply(states[active], rng.integers(0, self.n_moves, size=n))
        return states, ks

    def encode(self, states):
        B = states.shape[0]
        oh = np.zeros((B, self.n_in), dtype=np.float32)
        idx = np.arange(self.n_slots) * self.n_sym + states
        oh[np.arange(B)[:, None], idx] = 1.0
        return oh

    def is_solved(self, states):
        return (states == self.solved).all(axis=1)


# -------------------------------------------------------------------------- bfs
def bfs(task, chunk=1_000_000, verbose=True):
    """Exact distance to solved for every reachable state, by breadth-first search."""
    dist = np.full(task.cells, UNSEEN, dtype=np.uint8)
    frontier = task.solved[None, :].copy()
    dist[perm_rank(frontier)] = 0
    seen, levels = 1, [1]
    t0 = time.time()

    depth = 0
    while frontier.shape[0]:
        depth += 1
        found = []
        for lo in range(0, frontier.shape[0], chunk):
            part = frontier[lo:lo + chunk]
            kids = task.children(part).reshape(-1, task.n_slots)
            idx = perm_rank(kids)
            fresh = dist[idx] == UNSEEN
            if not fresh.any():
                continue
            idx, kids = idx[fresh], kids[fresh]
            # the same state can be reached twice inside one chunk
            uniq = np.unique(idx, return_index=True)[1]
            idx, kids = idx[uniq], kids[uniq]
            still = dist[idx] == UNSEEN
            idx, kids = idx[still], kids[still]
            dist[idx] = depth
            found.append(kids)

        frontier = (np.concatenate(found) if found
                    else np.empty((0, task.n_slots), np.int8))
        if frontier.shape[0]:
            seen += frontier.shape[0]
            levels.append(int(frontier.shape[0]))
            if verbose:
                print(f"  depth {depth:>2}  {frontier.shape[0]:>12,} new   "
                      f"{seen:>12,} total   {time.time()-t0:>6.0f}s", flush=True)

    return dist, levels, seen, depth - 1, time.time() - t0


# --------------------------------------------------------------------- selftest
# Published diameters, in single-tile moves, blank counted as part of the state.
# 3x3 is the one that matters: the 8-puzzle's 31 is a long-standing published
# result, so agreeing with it checks the move model, the ranking and the BFS at
# once, against a number this file cannot have influenced.
KNOWN_DIAMETER = {"tile-2x2": 6, "tile-2x3": 21, "tile-2x4": 36, "tile-3x3": 31}


def selftest():
    ok = True

    # 1. the ranking is a bijection on a small n
    n = 6
    all_ranks = np.arange(math.factorial(n))
    perms = perm_unrank(all_ranks, n)
    back = perm_rank(perms)
    bijective = np.array_equal(back, all_ranks) and len(np.unique(back)) == len(all_ranks)
    print(f"  perm_rank bijection on {n}! = {math.factorial(n)}  "
          f"{'OK' if bijective else '*** FAIL ***'}")
    ok &= bijective

    # 2. undoing a LEGAL move restores the state, and an illegal one is a no-op.
    # The legality split matters and is not pedantry: U applied where the blank
    # is already on the top row does nothing, so a following D slides the blank
    # down and the pair does NOT restore. An earlier version of this check
    # ignored that and failed against correct code.
    t = TileTask("tile-3x3")
    rng = np.random.default_rng(0)
    st, _ = t.scramble(4096, 40, rng)
    inverse = {"U": "D", "D": "U", "L": "R", "R": "L"}
    for m, tok in enumerate(t.tokens):
        mi = t.tokens.index(inverse[tok])
        legal = t.dest[t.blank_of(st), m] >= 0
        sub = st[legal]
        moved = t.apply(sub, np.full(sub.shape[0], m))
        undone = t.apply(moved, np.full(sub.shape[0], mi))
        restores = np.array_equal(undone, sub) and (moved != sub).any()
        stuck = st[~legal]
        noop = np.array_equal(t.apply(stuck, np.full(stuck.shape[0], m)), stuck)
        good = restores and noop
        print(f"  {tok}: {inverse[tok]} undoes it on {legal.sum():>5} legal, "
              f"no-op on {(~legal).sum():>5} illegal   "
              f"{'OK' if good else '*** FAIL ***'}")
        ok &= good

    # 3. every scrambled state is a genuine permutation
    perm_ok = all(len(np.unique(row)) == t.n_slots for row in st[:256])
    print(f"  scrambles are permutations   {'OK' if perm_ok else '*** FAIL ***'}")
    ok &= perm_ok

    # 4. exhaustive BFS reproduces n!/2 and the published diameter
    for name in ("tile-2x2", "tile-2x3", "tile-2x4", "tile-3x3"):
        tt = TileTask(name)
        dist, levels, seen, diameter, secs = bfs(tt, verbose=False)
        want_n, want_d = tt.size, KNOWN_DIAMETER[name]
        good = seen == want_n and diameter == want_d
        print(f"  {name}: {seen:>7,} states (want {want_n:,}), "
              f"diameter {diameter} (want {want_d})   "
              f"{'OK' if good else '*** FAIL ***'}   {secs:.1f}s")
        ok &= good

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="", help="RxC, e.g. 3x3")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--save-table", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    if not args.board:
        raise SystemExit("give --board RxC or --selftest")

    task = TileTask(f"tile-{args.board}")
    print(f"{task.name}: {task.size:,} reachable states, "
          f"{task.cells:,} index cells ({task.cells/1e6:.0f} MB)\n")

    dist, levels, seen, diameter, secs = bfs(task, args.chunk)

    ok = seen == task.size
    print(f"\n  reached {seen:,} of {task.size:,} expected   "
          f"{'OK' if ok else '*** MISMATCH ***'}")
    print(f"  diameter {diameter}   built in {secs:.0f}s")
    if not ok:
        raise SystemExit("reachable-set size disagrees with n!/2: bug, not a result")

    out = HERE.parent / "eval" / "results" / f"exact-{task.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": task.name, "rows": task.rows, "cols": task.cols,
        "states": seen, "diameter": diameter, "seconds": secs,
        "histogram": [{"depth": d, "count": n} for d, n in enumerate(levels)],
        "mean_distance": float(sum(d * n for d, n in enumerate(levels)) / seen),
    }, indent=2) + "\n")
    print(f"  -> {out.relative_to(HERE.parent)}")

    if args.save_table:
        np.save(HERE / f"exact_{task.name}.npy", dist)
        print(f"  -> exact_{task.name}.npy  ({dist.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
