"""Exact distance-to-solved for the small ladder rungs, by breadth-first search.

This is what makes the ladder a real experiment rather than a demo. On rungs
small enough to enumerate we know the TRUE optimal solution length for every
state, so we can measure the learned solver's optimality gap directly, the same
claim DeepCubeA makes on the 3x3, and then watch that claim become unverifiable
as the rungs grow. It also gives each rung's exact diameter, which controls for
the obvious confound: a larger space is usually also a deeper one, and without
the diameter we could not tell those two effects apart.

    ../.venv-mlx/bin/python exact.py --k 5
    ../.venv-mlx/bin/python exact.py --k 6        # ~190MB, tens of minutes

Indexing is mixed-radix base 24 over the tracked pieces' slot positions. That
wastes some space: 24^k cells for P(24,k) reachable states, about 2x at k=6, but it is a branch-free vectorised computation, which matters far more here than
the memory: a minimal ranking would need a sequential "how many unused slots are
below this one" loop that numpy cannot do in one shot.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from davi import Task, rung_size

HERE = Path(__file__).parent
UNSEEN = 255


def indexer(k):
    """-> f(states) giving a unique int64 per state, in [0, 24^k)."""
    radix = (24 ** np.arange(k)).astype(np.int64)

    def index(states):
        # states is (B, 24); find which slot holds each tracked piece
        pos = np.empty((states.shape[0], k), dtype=np.int64)
        for piece in range(k):
            pos[:, piece] = np.argmax(states == piece, axis=1)
        return pos @ radix

    return index


def bfs(k, chunk=200_000, verbose=True, moves="all"):
    task = Task(f"wings-k{k}", moves=moves)
    index = indexer(k)
    dist = np.full(24 ** k, UNSEEN, dtype=np.uint8)

    frontier = task.solved[None, :].copy()
    dist[index(frontier)] = 0
    seen = 1
    levels = [1]
    t0 = time.time()

    depth = 0
    while frontier.shape[0]:
        depth += 1
        found = []
        # Chunk the expansion. A whole frontier's children at k=6 is tens of
        # millions of states at once, which is a needless memory spike.
        for lo in range(0, frontier.shape[0], chunk):
            part = frontier[lo:lo + chunk]
            kids = task.children(part).reshape(-1, task.n_slots)
            idx = index(kids)
            fresh = dist[idx] == UNSEEN
            if not fresh.any():
                continue
            idx, kids = idx[fresh], kids[fresh]
            # the same state can appear twice within one chunk's children
            uniq = np.unique(idx, return_index=True)[1]
            idx, kids = idx[uniq], kids[uniq]
            still = dist[idx] == UNSEEN
            idx, kids = idx[still], kids[still]
            dist[idx] = depth
            found.append(kids)

        frontier = np.concatenate(found) if found else np.empty((0, 24), np.int8)
        if frontier.shape[0]:
            seen += frontier.shape[0]
            levels.append(int(frontier.shape[0]))
            if verbose:
                print(f"  depth {depth:>2}  {frontier.shape[0]:>12,} new   "
                      f"{seen:>12,} total   {time.time()-t0:>6.0f}s", flush=True)

    return dist, levels, seen, depth - 1, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--moves", default="all")
    ap.add_argument("--save-table", action="store_true",
                    help="also write the full distance table (large)")
    args = ap.parse_args()

    expect = rung_size(args.k)
    cells = 24 ** args.k
    print(f"rung k={args.k}: {expect:,} reachable states, {cells:,} index cells "
          f"({cells/1e6:.0f} MB)\n")

    dist, levels, seen, diameter, secs = bfs(args.k, args.chunk, moves=args.moves)

    ok = seen == expect
    print(f"\n  reached {seen:,} of {expect:,} expected   "
          f"{'OK' if ok else '*** MISMATCH ***'}")
    print(f"  diameter {diameter}   built in {secs:.0f}s")
    if not ok:
        raise SystemExit("reachable-set size disagrees with P(24,k): bug, not a result")

    # Both output names come from the task, so the writer and every reader agree
    # on where a rung's results live.
    task = Task(f"wings-k{args.k}", moves=args.moves)
    out = task.histogram_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": args.k, "moves": args.moves, "states": seen, "diameter": diameter, "seconds": secs,
        "histogram": [{"depth": d, "count": n} for d, n in enumerate(levels)],
        "mean_distance": float(sum(d * n for d, n in enumerate(levels)) / seen),
    }, indent=2) + "\n")
    print(f"  -> {out.relative_to(HERE.parent)}")

    if args.save_table:
        np.save(task.table_path(), dist)
        print(f"  -> {task.table_path().name}  ({dist.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
