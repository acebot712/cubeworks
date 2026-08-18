"""Solve the 4x4 centres with the learned cost-to-go, and grade it honestly.

Search is batched beam search guided by J(s): keep the most promising `width`
states, expand every successor, keep the best `width` again. DeepCubeA uses
batch weighted A* for the same reason beam search works here, the network is
far more useful evaluating thousands of states at once on the GPU than one at a
time.

What "graded honestly" means. A learned heuristic is not admissible, so nothing
here is proven optimal. The two things that CAN be checked are:

  1. Does J actually track distance? Scramble k moves and see whether J rises
     with k. A network that collapsed to a constant will show a flat line, and
     a falling training loss will not reveal that.
  2. Does it beat the incumbent? The shipped greedy solver spends ~55 moves on
     centres. Anything that does not beat that is not worth its weight.

    ../.venv-mlx/bin/python solve_centers.py --n 20 --width 2000
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from train_centers import (
    N_MOVES, N_SLOTS, PERMS, SOLVED, TOKENS, ValueNet,
    all_children, apply_moves, encode, is_solved, scramble,
)

HERE = Path(__file__).parent


def load_net(path, hidden=None):
    if hidden is None:
        blob0 = np.load(path)
        sizes = [blob0[k].shape[0] for k in sorted(blob0.files) if k.endswith('.weight') and 'layers' in k]
        hidden = tuple(sizes)
    net = ValueNet(hidden)
    mx.eval(net.parameters())
    blob = np.load(path)
    params = net.parameters()
    for key, val in blob.items():
        node = params
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[int(p)] if p.isdigit() else node[p]
        last = parts[-1]
        node[int(last) if last.isdigit() else last] = mx.array(val)
    net.update(params)
    mx.eval(net.parameters())
    return net


def evaluate(net, states, chunk=8192):
    out = np.empty(states.shape[0], dtype=np.float32)
    for i in range(0, states.shape[0], chunk):
        part = states[i:i + chunk]
        out[i:i + chunk] = np.array(net(mx.array(encode(part))), copy=False)
    return out


def beam_solve(net, state, width=2000, max_depth=60):
    """-> list of move tokens, or None."""
    beam = state[None, :].copy()
    paths = [[]]
    for _ in range(max_depth):
        kids = all_children(beam).reshape(-1, N_SLOTS)
        moves = np.tile(np.arange(N_MOVES), beam.shape[0])
        parents = np.repeat(np.arange(beam.shape[0]), N_MOVES)

        done = is_solved(kids)
        if done.any():
            i = int(np.flatnonzero(done)[0])
            return paths[parents[i]] + [TOKENS[moves[i]]]

        j = evaluate(net, kids)
        # de-duplicate: the same arrangement reached twice wastes beam width
        _, uniq = np.unique(kids, axis=0, return_index=True)
        order = uniq[np.argsort(j[uniq])][:width]
        beam = kids[order]
        paths = [paths[parents[i]] + [TOKENS[moves[i]]] for i in order]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "centers_value.npz"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--kmax", type=int, default=24)
    args = ap.parse_args()

    net = load_net(args.model)
    rng = np.random.default_rng(7)

    # --- check 1: does J track distance at all? ---
    print("does the value function track distance?")
    print("  scramble depth ->  mean J   (should rise, and flatten near the")
    print("                              true diameter of the sub-problem)")
    for k in (1, 2, 4, 6, 8, 10, 14, 18, 24):
        st = np.tile(SOLVED, (512, 1))
        for _ in range(k):
            st = apply_moves(st, rng.integers(0, N_MOVES, size=512))
        print(f"    k = {k:2d}            {evaluate(net, st).mean():6.2f}")

    # --- check 2: does it actually solve, and in how many moves? ---
    print(f"\nsolving {args.n} scrambled centre states (beam width {args.width})")
    lens, fails = [], 0
    for i in range(args.n):
        st = np.tile(SOLVED, (1, 1))[0]
        st = apply_moves(st[None, :], rng.integers(0, N_MOVES, size=1))[0]
        for _ in range(args.kmax):
            st = apply_moves(st[None, :], rng.integers(0, N_MOVES, size=1))[0]
        sol = beam_solve(net, st, args.width)
        if sol is None:
            fails += 1
            continue
        # verify by replay: never trust a solver's own word for it
        check = st[None, :].copy()
        for mv in sol:
            check = apply_moves(check, np.array([TOKENS.index(mv)]))
        assert is_solved(check)[0], "solver returned an invalid solution"
        lens.append(len(sol))

    if lens:
        print(f"\n  solved {len(lens)}/{args.n}   mean {np.mean(lens):.1f} moves   "
              f"median {int(np.median(lens))}   best {min(lens)}   worst {max(lens)}")
    if fails:
        print(f"  failed to solve {fails}")
    print("\n  incumbent greedy centres solver: ~55 moves")


if __name__ == "__main__":
    main()
