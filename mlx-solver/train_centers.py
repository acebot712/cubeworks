"""Deep Approximate Value Iteration for the 4x4 centres, in MLX.

Scope, and why this sub-problem. Full-cube DeepCubeA is a research project: the
3x3 version needed ~10 billion training states and days of multi-GPU time for a
space of depth 20 and branching 18, and the 4x4 has depth ~40, branching ~36 and
10^26 times more states. The CENTRES alone are 3.25e15 states at depth ~20 —
comfortably trainable on one machine, and worth 55 of the ~169 moves the current
solver spends.

The method is DeepCubeA's, not AlphaZero's. There is no self-play here because a
cube is a single-agent shortest-path problem, not a two-player game. What makes
the data free is the same either way: scrambling k moves from solved generates a
labelled state, endlessly.

    J(s) = 0                        if s is solved
    J(s) = min_a [ 1 + J_target(s') ]   otherwise

Targets come from a periodically-frozen copy of the network, which is what stops
the bootstrap from chasing its own tail. Scrambles are drawn from k = 1..K so
that easy states anchor the value function near solved and it propagates outward.

Honest note on what this can and cannot give you: a learned heuristic is NOT
admissible — nothing stops the network overestimating — so solutions found with
it are near-optimal, never proven optimal. That is the same trade DeepCubeA
makes, and it is unavoidable for a learned value function.

    ../.venv-mlx/bin/python train_centers.py --steps 3000
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

HERE = Path(__file__).parent
N_SLOTS = 24
N_FACES = 6

_tbl = json.loads((HERE / "center_perms.json").read_text())
TOKENS = _tbl["tokens"]
PERMS = np.array(_tbl["perms"], dtype=np.int32)      # (36, 24)
N_MOVES = len(TOKENS)

SOLVED = np.repeat(np.arange(N_FACES, dtype=np.int8), 4)   # 4 of each face


def apply_moves(states, move_ids):
    """states (B, 24) int8, move_ids (B,) -> permuted copies."""
    out = np.empty_like(states)
    for m in range(N_MOVES):
        sel = move_ids == m
        if not sel.any():
            continue
        p = PERMS[m]
        out[np.ix_(sel, p)] = states[sel]
    return out


def all_children(states):
    """(B, 24) -> (B, N_MOVES, 24): every state's successors."""
    B = states.shape[0]
    out = np.empty((B, N_MOVES, N_SLOTS), dtype=np.int8)
    for m in range(N_MOVES):
        out[:, m, :][:, PERMS[m]] = states
    return out


def scramble(batch, k_max, rng):
    """Free training data: k random moves from solved, k ~ U(1, k_max)."""
    states = np.tile(SOLVED, (batch, 1))
    ks = rng.integers(1, k_max + 1, size=batch)
    for step in range(k_max):
        active = ks > step
        if not active.any():
            break
        mv = rng.integers(0, N_MOVES, size=active.sum())
        states[active] = apply_moves(states[active], mv)
    return states, ks


def encode(states):
    """(B, 24) -> (B, 144) one-hot. Which FACE sits in each slot."""
    B = states.shape[0]
    oh = np.zeros((B, N_SLOTS * N_FACES), dtype=np.float32)
    idx = np.arange(N_SLOTS) * N_FACES + states
    oh[np.arange(B)[:, None], idx] = 1.0
    return oh


def is_solved(states):
    return (states == SOLVED).all(axis=1)


class ValueNet(nn.Module):
    """Cost-to-go estimator. Deliberately small — the centres sub-problem does
    not need DeepCubeA's residual tower, and a model that trains in minutes is
    worth more here than one that trains in days."""

    def __init__(self, hidden=(1024, 512, 256)):
        super().__init__()
        dims = (N_SLOTS * N_FACES,) + hidden
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.head = nn.Linear(hidden[-1], 1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.head(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sync-every", type=int, default=100, help="target-network refresh")
    ap.add_argument("--hidden", default="1024,512,256",
                    help="layer sizes; small nets are for the JS port, where a "
                         "beam step cannot afford 800k params per state")
    ap.add_argument("--out", default=str(HERE / "centers_value.npz"))
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    rng = np.random.default_rng(0)
    net = ValueNet(hidden)
    target = ValueNet(hidden)
    mx.eval(net.parameters(), target.parameters())
    target.update(net.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, x, y):
        return mx.mean(mx.square(model(x) - y))

    grad_fn = nn.value_and_grad(net, loss_fn)
    t0 = time.time()
    running = None

    for step in range(1, args.steps + 1):
        states, _ = scramble(args.batch, args.kmax, rng)

        # target = min over successors of (1 + J_target(child)), 0 if solved
        kids = all_children(states)                       # (B, M, 24)
        flat = kids.reshape(-1, N_SLOTS)
        jt = np.array(target(mx.array(encode(flat))), copy=False).reshape(args.batch, N_MOVES)
        jt = np.maximum(jt, 0.0)
        solved_kids = is_solved(flat).reshape(args.batch, N_MOVES)
        jt[solved_kids] = 0.0
        y = (1.0 + jt).min(axis=1)
        y[is_solved(states)] = 0.0

        loss, grads = grad_fn(net, mx.array(encode(states)), mx.array(y.astype(np.float32)))
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)

        lv = float(loss)
        running = lv if running is None else 0.98 * running + 0.02 * lv
        if step % args.sync_every == 0:
            target.update(net.parameters())
            mx.eval(target.parameters())
            print(f"  step {step:5d}  loss {running:7.4f}  ({time.time()-t0:5.1f}s)", flush=True)

    flat_params = {}

    def flatten(prefix, tree):
        if isinstance(tree, dict):
            for k, v in tree.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                flatten(f"{prefix}.{i}", v)
        else:
            flat_params[prefix] = np.array(tree, copy=False)

    flatten("", net.parameters())
    np.savez(args.out, **flat_params)
    print(f"\nsaved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
