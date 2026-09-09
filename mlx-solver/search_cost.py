"""Greedy best-first search, counting expansions, as the cost side of E5.

The design and every analysis choice are fixed in
`eval/PREREGISTRATION-search-cost.md`, written before this file ran. Read that
first; this module only implements the search it specifies.

WHY GREEDY BEST-FIRST AND NOT BEAM. Under beam search at fixed width the
expansion count is `width x branching x levels`, which is solution length in
disguise with almost no dynamic range. Greedy best-first has no width parameter,
its expansion count genuinely varies with heuristic quality, and it is the
algorithm Wilt and Ruml measured, so the comparison against the prior work is
like for like.

WHY THERE IS NO CENSORING. Every reachable state can reach every other, and the
closed list stops a state being expanded twice, so the search cannot expand more
than `|S|` nodes and always reaches the goal. Cost is defined for every
(heuristic, instance) pair, and the censored-data estimator the paper's
Limitations section proposed is not needed. This is why the pre-registration
excludes `wings-k5` and `wings-k6`: the bound stops being affordable there.

WHY THE SEARCHES RUN INTERLEAVED. A learned heuristic evaluated on one node's
four children is a batch of four, which is far too small for the GPU, and the run
becomes overhead-bound at roughly 2k expansions per second. Advancing many
independent searches by one expansion each puts every instance's children into
ONE call, and the same work takes about 1/1000th the wall clock. The searches do
not interact: each keeps its own open heap and its own closed list, and
interleaving changes only when the heuristic is evaluated, never what is
expanded.

Cost is capped by EXPANSIONS and never by wall clock. A time cap would
systematically favour pattern databases, an array lookup, over learned
heuristics, a forward pass, and would manufacture exactly the learned versus
classical difference this project spent five controls ruling out.

    ../.venv-mlx/bin/python search_cost.py --selftest
"""
import argparse
import heapq
import sys

import numpy as np

from domains import (abstract, cells_of, load_table, make_task, project,
                     rank, rungs)


def pdb_heuristic(task, j):
    """Rung j of `task` as an admissible heuristic: one table lookup per state."""
    sub = abstract(task, j)
    tab = load_table(sub)

    def h(states):
        return tab[rank(sub, project(task, states, j))].astype(np.float32)
    return h


def exact_heuristic(task):
    """The true distance. Not a baseline, only the self-test's oracle."""
    tab = load_table(task)

    def h(states):
        return tab[rank(task, states)].astype(np.float32)
    return h


def random_heuristic(seed=0):
    rng = np.random.default_rng(seed)

    def h(states):
        return rng.random(states.shape[0]).astype(np.float32)
    return h


def greedy_expansions(task, h_fn, starts, cap=None, chunk=250, verbose=False):
    """Expansions to goal, and solution length, for each start state.

    Returns (expansions, lengths), both int64 arrays aligned with `starts`.

    An expansion is counted when a node is removed from the open list, INCLUDING
    the one on which the goal is detected. So a perfect heuristic costs d+1 on a
    start at distance d, not d. The convention only has to be constant across
    the heuristics being compared, and this is the one the self-test pins.

    `chunk` is how many searches are interleaved at once. It trades memory, one
    closed list of `task.cells` bools per live search, against heuristic batch
    size. It does not affect the result: each search's open heap and closed list
    are private, so the node it expands next depends only on its own state.
    """
    starts = np.asarray(starts, dtype=np.int8)
    n = starts.shape[0]
    cells = cells_of(task)
    cap = cap if cap is not None else cells
    expansions = np.zeros(n, dtype=np.int64)
    lengths = np.full(n, -1, dtype=np.int64)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ids = list(range(lo, hi))
        # One private heap and one private closed list per search. Entries are
        # (h, tiebreak, g, state_bytes); the counter keeps ordering total so
        # heapq never compares two state buffers.
        heaps = {i: [] for i in ids}
        closed = {i: np.zeros(cells, dtype=bool) for i in ids}
        tick = 0

        h0 = h_fn(starts[lo:hi])
        for k, i in enumerate(ids):
            s = starts[i]
            closed[i][int(rank(task, s[None, :])[0])] = True
            heapq.heappush(heaps[i], (float(h0[k]), tick, 0, s.tobytes()))
            tick += 1

        live = [i for i in ids if not task.is_solved(starts[i][None, :])[0]]
        for i in ids:
            if i not in live:
                lengths[i] = 0                      # already solved

        while live:
            popped, popped_g, still = [], [], []
            for i in live:
                if not heaps[i] or expansions[i] >= cap:
                    lengths[i] = -1                 # exhausted; see module docstring
                    continue
                _, _, g, sb = heapq.heappop(heaps[i])
                popped.append(np.frombuffer(sb, dtype=np.int8))
                popped_g.append(g)
                still.append(i)
            if not still:
                break

            states = np.stack(popped)
            gs = np.array(popped_g, dtype=np.int64)
            expansions[still] += 1

            done = task.is_solved(states)
            if done.any():
                for k in np.flatnonzero(done):
                    lengths[still[int(k)]] = int(gs[int(k)])
                keep = ~done
                states, gs = states[keep], gs[keep]
                still = [i for k, i in enumerate(still) if keep[k]]
                if not still:
                    live = []
                    break

            kids = task.children(states)                       # (B, b, slots)
            b = kids.shape[1]
            flat = kids.reshape(-1, task.n_slots)
            owner = np.repeat(np.array(still), b)
            kid_g = np.repeat(gs + 1, b)
            idx = rank(task, flat)

            # Drop anything already closed for ITS OWN search, then close it, so
            # a state generated twice in one round is only pushed once.
            fresh = np.empty(flat.shape[0], dtype=bool)
            for k, i in enumerate(still):
                sl = slice(k * b, (k + 1) * b)
                seen = closed[i][idx[sl]]
                fresh[sl] = ~seen
                closed[i][idx[sl][~seen]] = True
            if not fresh.any():
                live = [i for i in still if heaps[i]]
                continue

            flat, owner, kid_g = flat[fresh], owner[fresh], kid_g[fresh]
            hv = h_fn(flat)                                    # ONE batched call
            for k in range(flat.shape[0]):
                i = int(owner[k])
                heapq.heappush(heaps[i], (float(hv[k]), tick, int(kid_g[k]),
                                          flat[k].tobytes()))
                tick += 1

            live = [i for i in still if heaps[i] and expansions[i] < cap]
            if verbose and tick % 50000 < b:
                print(f"    {len(live)} live, max expansions "
                      f"{expansions[lo:hi].max():,}", flush=True)

        for i in ids:
            closed[i] = None
    return expansions, lengths


# ------------------------------------------------------------------- selftest
def selftest():
    ok = True
    for name in ("tile-2x3", "tile-3x3", "wings-k4"):
        task = make_task(name)
        exact = load_table(task)
        rng = np.random.default_rng(0)
        starts, _ = task.scramble(60, 40, rng)
        true_d = exact[rank(task, starts)].astype(np.int64)

        # 1. With the exact distance as the heuristic, greedy best-first walks
        # straight down the gradient: the open list's minimum is always the next
        # state on an optimal path, so it returns a length-d solution having
        # expanded exactly d+1 nodes, the goal included. Any other answer means
        # the search, the closed list or the goal test is wrong.
        e, L = greedy_expansions(task, exact_heuristic(task), starts, chunk=30)
        want_e = np.where(true_d == 0, 0, true_d + 1)
        opt = bool(np.array_equal(L, true_d))
        cheap = bool(np.array_equal(e, want_e))
        print(f"  {name:>9} oracle heuristic: length == d {'OK' if opt else '*** FAIL ***'}"
              f"   expansions == d+1 {'OK' if cheap else '*** FAIL ***'}"
              f"  (max d {true_d.max()})")
        ok &= opt and cheap

        # 2. A real abstraction solves everything, costs more than the oracle,
        # and never returns a path shorter than optimal.
        js = rungs(task)
        if js:
            e2, L2 = greedy_expansions(task, pdb_heuristic(task, js[-1]), starts, chunk=30)
            solved = bool((L2 >= 0).all())
            dearer = bool((e2 >= e).all())
            sound = bool((L2 >= true_d).all())
            print(f"  {' ':>9} PDB rung {js[-1]}: all solved {'OK' if solved else '*** FAIL ***'}"
                  f"   costs >= oracle {'OK' if dearer else '*** FAIL ***'}"
                  f"   no sub-optimal-length path {'OK' if sound else '*** FAIL ***'}"
                  f"   median expansions {int(np.median(e2))}")
            ok &= solved and dearer and sound

    # 3. Interleaving must not change the answer: chunk is a batching knob, not
    # a search parameter.
    task = make_task("tile-3x3")
    rng = np.random.default_rng(1)
    starts, _ = task.scramble(40, 40, rng)
    j = rungs(task)[2]
    a, _ = greedy_expansions(task, pdb_heuristic(task, j), starts, chunk=1)
    b, _ = greedy_expansions(task, pdb_heuristic(task, j), starts, chunk=40)
    same = bool(np.array_equal(a, b))
    print(f"  {'':>9} chunk=1 and chunk=40 agree {'OK' if same else '*** FAIL ***'}")
    ok &= same

    print("\n  ALL CHECKS PASSED" if ok else "\n  *** SELFTEST FAILED ***")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    raise SystemExit("nothing to do yet; the experiment driver is not written")


if __name__ == "__main__":
    main()
