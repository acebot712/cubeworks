"""Measure one trained checkpoint, with the statistics a reviewer will ask for.

Reports, as JSON:

  solve rate        with a Wilson score interval, not "7/10". A binomial point
                    estimate from a small n is not a result, and Wilson is the
                    right interval for proportions near 0 or 1 where the normal
                    approximation is worst.
  optimality gap    only on rungs where exact.py has enumerated the ground truth.
                    This is the strong claim: how often the learned solver finds
                    a genuinely shortest path, and it is available precisely
                    where the space is small enough to verify, which is the honest
                    limit of the whole approach.
  saturation        J at a range of scramble depths, plus the slope over the deep
                    band. A heuristic that has run out of resolution far from the
                    goal shows a slope near zero there while looking fine near it.

    ../.venv-mlx/bin/python evaluate.py --task wings-k6 --n 200 --width 100
"""
import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from davi import Task, ValueNet, read_ckpt
from domains import load_table, make_task, missing_table

HERE = Path(__file__).parent
RESULTS = HERE.parent / "eval" / "results"


def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a proportion. Degrades gracefully at 0/n and n/n,
    which is exactly where our small-sample results live."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def load(task_name, tag=""):
    stem = f"{task_name}{tag}"
    meta = json.loads((HERE / f"ckpt_{stem}.json").read_text())
    task = make_task(task_name, moves=meta.get("moves", "all"))
    net = ValueNet(task.n_in, tuple(meta["hidden"]))
    mx.eval(net.parameters())
    net.update(read_ckpt(np.load(HERE / f"ckpt_{stem}.npz"), "net.") or {})
    mx.eval(net.parameters())
    return task, net, meta


def j_of(task, net, states, chunk=16384):
    out = np.empty(states.shape[0], dtype=np.float32)
    for i in range(0, states.shape[0], chunk):
        part = states[i:i + chunk]
        out[i:i + chunk] = np.array(net(mx.array(task.encode(part))), copy=False)
    return out


def beam_solve(task, net, state, width, max_depth):
    beam = state[None, :].copy()
    paths = [[]]
    for _ in range(max_depth):
        kids = task.children(beam).reshape(-1, task.n_slots)
        moves = np.tile(np.arange(task.n_moves), beam.shape[0])
        parents = np.repeat(np.arange(beam.shape[0]), task.n_moves)
        done = task.is_solved(kids)
        if done.any():
            i = int(np.flatnonzero(done)[0])
            return paths[parents[i]] + [task.tokens[moves[i]]]
        j = j_of(task, net, kids)
        _, uniq = np.unique(kids, axis=0, return_index=True)
        order = uniq[np.argsort(j[uniq])][:width]
        beam = kids[order]
        paths = [paths[parents[i]] + [task.tokens[moves[i]]] for i in order]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--scramble", type=int, default=0, help="0 = the rung's training kmax")
    ap.add_argument("--max-depth", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234, help="evaluation scrambles, held fixed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task, net, meta = load(args.task, args.tag)
    kmax = args.scramble or meta.get("kmax", 40)
    rng = np.random.default_rng(args.seed)

    # --- saturation ---------------------------------------------------------
    depths = [d for d in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 26, 32, 40) if d <= kmax]
    curve = []
    for d in depths:
        st = np.tile(task.solved, (512, 1))
        for _ in range(d):
            st = task.apply(st, rng.integers(0, task.n_moves, size=512))
        curve.append((d, float(j_of(task, net, st).mean())))

    # slope over the deepest half: near zero means the heuristic cannot tell
    # far-from-goal states apart, which is invisible in the loss
    deep = [(d, j) for d, j in curve if d >= max(depths) / 2]
    slope = ((deep[-1][1] - deep[0][1]) / (deep[-1][0] - deep[0][0])) if len(deep) > 1 else 0.0

    # --- ground truth, where it exists --------------------------------------
    # The task says where its table lives. Composing the name here meant
    # composing the CUBE's name, and a board's rung number lives in the cube's
    # rung namespace: handed tile-3x3 this asked for exact_k8.npy, a cube rung's
    # table rather than an absent one. The optimality block then vanished in
    # silence, so a sliding-tile evaluation reported no gap and never said why.
    no_table = missing_table(task)
    exact = None if no_table else load_table(task)

    # --- solving ------------------------------------------------------------
    lens, opts, solved, t0 = [], [], 0, time.time()
    for i in range(args.n):
        st = np.tile(task.solved, (1, 1))
        for _ in range(kmax):
            st = task.apply(st, rng.integers(0, task.n_moves, size=1))
        sol = beam_solve(task, net, st[0], args.width, args.max_depth)
        if sol is None:
            continue
        check = st.copy()
        for mv in sol:
            check = task.apply(check, np.array([task.tokens.index(mv)]))
        assert task.is_solved(check)[0], "solver returned an invalid solution"
        solved += 1
        lens.append(len(sol))
        if exact is not None:
            true_d = int(exact[task.rank(st)[0]])
            # A solution shorter than the shortest is impossible, so a negative
            # excess means the table being read is not this task's, or the cell
            # is one BFS never reached and still holds its sentinel. Either way
            # the number would be silently wrong rather than absent.
            assert len(sol) >= true_d, (
                f"{task.name} read {true_d} from {task.table_path().name} for a "
                f"state solved in {len(sol)}: wrong table, or an unreached cell")
            opts.append(len(sol) - true_d)

    p, lo, hi = wilson(solved, args.n)
    res = {
        "task": args.task, "tag": args.tag, "k": task.k, "states": task.size,
        "step": meta.get("step"), "hidden": meta.get("hidden"), "kmax": kmax,
        "width": args.width, "n": args.n,
        "solved": solved, "solve_rate": p, "ci95": [lo, hi],
        "mean_len": float(np.mean(lens)) if lens else None,
        "j_curve": curve, "deep_slope": slope,
        "seconds_per_state": (time.time() - t0) / args.n,
    }
    if opts:
        res["optimality"] = {
            "mean_excess": float(np.mean(opts)),
            "exact_rate": float(np.mean([o == 0 for o in opts])),
            "worst_excess": int(max(opts)),
        }
    elif no_table:
        # An absent optimality block used to be indistinguishable from one that
        # was never asked for. Recording the reason means a reader of the JSON,
        # or of the console, is told which table is missing rather than left to
        # infer it from a key that is not there.
        res["optimality_unavailable"] = no_table

    print(f"{args.task}{args.tag}  states {task.size:.2e}  step {res['step']:,}")
    print(f"  solved {solved}/{args.n} = {p*100:.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f}%)")
    if lens:
        print(f"  mean solution {np.mean(lens):.2f} moves")
    if opts:
        o = res["optimality"]
        print(f"  OPTIMAL {o['exact_rate']*100:.1f}% of solves   "
              f"mean excess {o['mean_excess']:+.2f}   worst {o['worst_excess']:+d}")
    elif no_table:
        print(f"  no optimality gap: {no_table}")
    print(f"  J deep-band slope {slope:+.4f} per move   "
          f"J({depths[0]})={curve[0][1]:.2f} -> J({depths[-1]})={curve[-1][1]:.2f}")

    out = Path(args.out) if args.out else RESULTS / f"eval-{args.task}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
