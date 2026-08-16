"""Search with a trained cost-to-go network, and grade it honestly.

Works on either sub-problem's checkpoint:

    ../.venv-mlx/bin/python solve.py --task centers --n 20 --width 300
    ../.venv-mlx/bin/python solve.py --task wings   --n 20 --width 2000

Batched beam search: keep the `width` most promising states, expand every
successor, keep the best `width` again. DeepCubeA uses batch weighted A* for the
same underlying reason beam search works here — the network is far more useful
evaluating thousands of states at once on the GPU than one at a time.

Two things can be checked, and both are:

  1. Does J track distance? (`--probe`, or the table printed below.) A network
     that collapsed to a constant shows a flat line, and a falling training loss
     will not reveal it.
  2. Does it solve, and in how many moves? Every solution is replayed on the
     puzzle before it is counted. A solver's own word is not evidence.

What is NOT checked, because it cannot be: optimality. A learned heuristic is
not admissible, so these solutions are near-optimal and never proven minimal.
"""
import argparse
import time

import mlx.core as mx
import numpy as np

from davi import HERE, Task, ValueNet, probe, read_ckpt


def load(task_name, path=None, hidden=None):
    import json
    ckpt = path or (HERE / f"ckpt_{task_name}.npz")
    meta_path = HERE / f"ckpt_{task_name}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hidden = tuple(hidden or meta.get("hidden") or (256, 128))
    task = Task(task_name)
    net = ValueNet(task.n_in, hidden)
    mx.eval(net.parameters())
    net.update(read_ckpt(np.load(ckpt), "net.") or {})
    mx.eval(net.parameters())
    return task, net, meta


def evaluate(task, net, states, chunk=16384):
    out = np.empty(states.shape[0], dtype=np.float32)
    for i in range(0, states.shape[0], chunk):
        part = states[i:i + chunk]
        out[i:i + chunk] = np.array(net(mx.array(task.encode(part))), copy=False)
    return out


def beam_solve(task, net, state, width=1000, max_depth=60):
    """-> list of move tokens, or None."""
    beam = state[None, :].copy()
    paths = [[]]
    nodes = 0
    for _ in range(max_depth):
        kids = task.children(beam).reshape(-1, task.n_slots)
        moves = np.tile(np.arange(task.n_moves), beam.shape[0])
        parents = np.repeat(np.arange(beam.shape[0]), task.n_moves)
        nodes += kids.shape[0]

        done = task.is_solved(kids)
        if done.any():
            i = int(np.flatnonzero(done)[0])
            return paths[parents[i]] + [task.tokens[moves[i]]], nodes

        j = evaluate(task, net, kids)
        # de-duplicate: the same arrangement reached twice wastes beam width
        _, uniq = np.unique(kids, axis=0, return_index=True)
        order = uniq[np.argsort(j[uniq])][:width]
        beam = kids[order]
        paths = [paths[parents[i]] + [task.tokens[moves[i]]] for i in order]
    return None, nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["centers", "wings"], default="centers")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--scramble", type=int, default=0, help="0 = the task's training kmax")
    ap.add_argument("--max-depth", type=int, default=60)
    args = ap.parse_args()

    task, net, meta = load(args.task, args.model)
    kmax = args.scramble or meta.get("kmax", 24)
    rng = np.random.default_rng(7)

    print(f"{args.task}: checkpoint at step {meta.get('step', '?'):,}, "
          f"hidden {tuple(meta.get('hidden', ()))}, curriculum reached k={meta.get('k_cur', '?')}\n")

    print("does the value function track distance?")
    print("  (J should rise with k, then flatten near the sub-problem's diameter)")
    for k, j in probe(task, net, rng):
        print(f"    k = {k:2d}    J = {j:6.2f}")

    print(f"\nsolving {args.n} states scrambled {kmax} moves, beam width {args.width}")
    lens, fails, nodes, t0 = [], 0, 0, time.time()
    for i in range(args.n):
        st = np.tile(task.solved, (1, 1))
        for _ in range(kmax):
            st = task.apply(st, rng.integers(0, task.n_moves, size=1))
        sol, n = beam_solve(task, net, st[0], args.width, args.max_depth)
        nodes += n
        if sol is None:
            fails += 1
            print(f"  {i + 1:3d}  no solution within depth {args.max_depth}", flush=True)
            continue
        # replay it — never trust the solver's own word
        check = st.copy()
        for mv in sol:
            check = task.apply(check, np.array([task.tokens.index(mv)]))
        assert task.is_solved(check)[0], "solver returned an invalid solution"
        lens.append(len(sol))
        print(f"  {i + 1:3d}  {len(sol):3d} moves  (verified)", flush=True)

    el = time.time() - t0
    if lens:
        print(f"\n  solved {len(lens)}/{args.n}   mean {np.mean(lens):.1f}   "
              f"median {int(np.median(lens))}   best {min(lens)}   worst {max(lens)}")
    if fails:
        print(f"  failed {fails}/{args.n}")
    print(f"  {el / args.n:.1f}s and {nodes / args.n / 1000:.0f}k nodes per state")


if __name__ == "__main__":
    main()
