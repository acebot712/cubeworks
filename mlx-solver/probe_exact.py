"""Exact distance-to-solved for individual states on rungs too big to enumerate.

exact.py enumerates a whole rung and stops working around k=6 (96.9M states).
The rungs where the learned heuristic actually fails start at k=8, so ground
truth ran out exactly where it became interesting. That is not bad luck; it is
the same wall the whole field hits, and it is why papers in this area report
solve rates rather than optimality gaps on large problems.

Meet-in-the-middle gets past it for a SAMPLE of states, which is all a
measurement needs. Build the ball of radius f around solved once, then for each
query state walk outward b levels and look for an intersection. Any meeting
point gives a path through it, so

    distance(s) = min over meetings of (forward depth + backward depth)

and iterating b upward lets us stop as soon as b exceeds the best total found, no further level can beat it. Reaching depth f+b costs storage that grows with
f only, and per-query time that grows with b only, so the split is a knob:
a big one-time table buys cheap queries.

Every move set here is closed under inverses (U' is present whenever U is), so
walking outward from the query state is the same as walking backward towards it,
and no inverse permutation table is needed.

    # verify against exhaustive ground truth first: always
    ../.venv-mlx/bin/python probe_exact.py --k 6 --verify 300

    # then use it where enumeration cannot reach
    ../.venv-mlx/bin/python probe_exact.py --k 8 --n 2000 --forward 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from davi import Task
from exact import indexer

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"


def build_forward(task, index, depth, cap, verbose=True):
    """Ball of radius `depth` around solved -> (sorted indices, their distances)."""
    seen_idx = [np.array([index(task.solved[None, :])[0]], dtype=np.int64)]
    seen_d = [np.zeros(1, dtype=np.uint8)]
    known = seen_idx[0].copy()
    frontier = task.solved[None, :].copy()
    t0 = time.time()

    for d in range(1, depth + 1):
        kids = task.children(frontier).reshape(-1, task.n_slots)
        idx = index(kids)
        # drop duplicates within this level, then anything already reached
        uniq = np.unique(idx, return_index=True)[1]
        idx, kids = idx[uniq], kids[uniq]
        fresh = ~np.isin(idx, known, assume_unique=False)
        idx, kids = idx[fresh], kids[fresh]
        if not idx.size:
            break
        seen_idx.append(idx)
        seen_d.append(np.full(idx.size, d, dtype=np.uint8))
        known = np.union1d(known, idx)
        frontier = kids
        if verbose:
            print(f"    forward depth {d:>2}  {idx.size:>12,} new  "
                  f"{known.size:>12,} total  {time.time()-t0:>5.0f}s  "
                  f"({100*known.size/cap:.0f}% of cap)", flush=True)
        if known.size > cap:
            print(f"    stopping: ball exceeds --cap ({cap:,})", flush=True)
            depth = d
            break

    all_idx = np.concatenate(seen_idx)
    all_d = np.concatenate(seen_d)
    order = np.argsort(all_idx, kind="stable")
    all_idx, all_d = all_idx[order], all_d[order]
    # a state can be reached at several depths; keep the smallest
    keep = np.ones(all_idx.size, dtype=bool)
    keep[1:] = all_idx[1:] != all_idx[:-1]
    return all_idx[keep], all_d[keep], depth


def lookup(table_idx, table_d, idx):
    """-> distances for the entries present in the table, else -1."""
    pos = np.searchsorted(table_idx, idx)
    pos = np.clip(pos, 0, table_idx.size - 1)
    hit = table_idx[pos] == idx
    out = np.full(idx.size, -1, dtype=np.int32)
    out[hit] = table_d[pos[hit]]
    return out


def distance(task, index, table_idx, table_d, state, max_back):
    """Exact distance from `state` to solved, or None if beyond f + max_back."""
    best = None
    frontier = state[None, :].copy()
    seen = index(frontier)

    d0 = lookup(table_idx, table_d, seen)
    if d0[0] >= 0:
        best = int(d0[0])
        if best == 0:
            return 0

    for b in range(1, max_back + 1):
        if best is not None and b >= best:
            break                      # any hit from here on totals >= b >= best
        kids = task.children(frontier).reshape(-1, task.n_slots)
        idx = index(kids)
        uniq = np.unique(idx, return_index=True)[1]
        idx, kids = idx[uniq], kids[uniq]
        fresh = ~np.isin(idx, seen)
        idx, kids = idx[fresh], kids[fresh]
        if not idx.size:
            break
        d = lookup(table_idx, table_d, idx)
        got = d >= 0
        if got.any():
            cand = int(d[got].min()) + b
            best = cand if best is None else min(best, cand)
        seen = np.union1d(seen, idx)
        frontier = kids
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--moves", default="all")
    ap.add_argument("--forward", type=int, default=5, help="radius of the stored ball")
    ap.add_argument("--back", type=int, default=6, help="max levels walked per query")
    ap.add_argument("--cap", type=int, default=40_000_000, help="max states in the ball")
    ap.add_argument("--n", type=int, default=1000, help="states to probe")
    ap.add_argument("--scramble", type=int, default=30)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--verify", type=int, default=0,
                    help="cross-check this many states against exact.py's table")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task = Task(f"wings-k{args.k}", moves=args.moves)
    index = indexer(args.k)
    rng = np.random.default_rng(args.seed)

    print(f"rung k={args.k} ({args.moves}, {task.n_moves} moves), "
          f"{task.size:.3e} states: enumeration "
          f"{'feasible' if task.size < 3e8 else 'INFEASIBLE'}, meeting in the middle\n")

    t0 = time.time()
    table_idx, table_d, f_used = build_forward(task, index, args.forward, args.cap)
    print(f"  ball of radius {f_used}: {table_idx.size:,} states, "
          f"{table_idx.nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s")
    print(f"  reach: exact for distances up to {f_used + args.back}\n")

    # --- correctness: the prober must agree with exhaustive BFS where both run --
    if args.verify:
        suf = "" if args.moves == "all" else f"-{args.moves}"
        path = HERE / f"exact_k{args.k}{suf}.npy"
        if not path.exists():
            raise SystemExit(f"--verify needs {path.name}; run exact.py --k {args.k} "
                             f"--moves {args.moves} --save-table")
        truth = np.load(path)
        bad = 0
        st = np.tile(task.solved, (args.verify, 1))
        for _ in range(args.scramble):
            st = task.apply(st, rng.integers(0, task.n_moves, size=args.verify))
        for i in range(args.verify):
            got = distance(task, index, table_idx, table_d, st[i], args.back)
            want = int(truth[index(st[i:i + 1])[0]])
            if got != want:
                bad += 1
                if bad <= 5:
                    print(f"    MISMATCH: probe {got} vs exact {want}")
        print(f"  verify: {args.verify - bad}/{args.verify} agree with exhaustive BFS"
              f"  {'OK' if bad == 0 else '*** BROKEN ***'}\n")
        if bad:
            raise SystemExit("prober disagrees with ground truth: do not use")

    # --- probe ---------------------------------------------------------------
    st = np.tile(task.solved, (args.n, 1))
    for _ in range(args.scramble):
        st = task.apply(st, rng.integers(0, task.n_moves, size=args.n))

    t1 = time.time()
    dists, unresolved = [], 0
    for i in range(args.n):
        d = distance(task, index, table_idx, table_d, st[i], args.back)
        if d is None:
            unresolved += 1
        else:
            dists.append(d)
    el = time.time() - t1

    print(f"  probed {args.n} states scrambled {args.scramble} moves in {el:.0f}s "
          f"({el/args.n*1000:.0f} ms each)")
    if dists:
        u, c = np.unique(dists, return_counts=True)
        print(f"  exact distances: mean {np.mean(dists):.2f}  "
              f"min {min(dists)}  max {max(dists)}")
        print("  " + "  ".join(f"d={a}:{b}" for a, b in zip(u, c)))
    if unresolved:
        print(f"  {unresolved} beyond reach {f_used + args.back}: raise --forward or --back")

    out = Path(args.out) if args.out else \
        RESULTS / f"probe-k{args.k}{'' if args.moves=='all' else '-'+args.moves}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": args.k, "moves": args.moves, "states": task.size,
        "forward": f_used, "back": args.back, "reach": f_used + args.back,
        "ball_size": int(table_idx.size), "n": args.n, "scramble": args.scramble,
        "seed": args.seed, "unresolved": unresolved,
        "ms_per_state": el / args.n * 1000,
        "distances": [int(d) for d in dists],
    }, indent=2) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
