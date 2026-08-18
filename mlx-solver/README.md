# Learned solvers

Deep Approximate Value Iteration on the 4×4's sub-problems, in MLX on the Mac's
GPU. The method is DeepCubeA's, not AlphaZero's: a cube is a single-agent
shortest-path problem, so there is no adversary and no self-play. What makes the
data free is the same either way: scrambling *k* moves from solved produces a
labelled state, endlessly, with no dataset to collect and nothing to annotate.

```
J(s) = 0                             if s is solved
J(s) = min over moves of 1 + J_target(s')   otherwise
```

`J_target` is a periodically-frozen copy of the network, which is what stops the
bootstrap from chasing its own tail.

## Running

```bash
npm run train -- --task centers --steps 8000       # minutes
npm run train -- --task wings   --steps 150000     # hours; resumes if interrupted
```

Progress prints a live rate and ETA. Every `--ckpt-every` steps it writes
`ckpt_<task>.npz` (network **and** Adam moments) plus a one-line entry in
`metrics_<task>.jsonl`. Re-running the same command picks up where it stopped: crash, closed lid, or Ctrl-C. `--fresh` starts over.

Evaluate any checkpoint:

```bash
../.venv-mlx/bin/python solve.py --task wings --n 20 --width 1000
```

Every solution is replayed on the puzzle before it is counted. A solver's own
word is not evidence.

## The two sub-problems

|                    | centres        | edge wings     |
| ------------------ | -------------- | -------------- |
| states             | 3.25 × 10¹⁵    | 3.10 × 10²³    |
| moves              | 36             | 63             |
| encoding           | 24 × 6 = 144   | 24 × 24 = 576  |
| default network    | 256, 128       | 4096, 2048, 1024 |
| curriculum         | off            | on             |
| target-net dtype   | fp32           | fp16           |

The wings are ~95 million times the centres. That gap is why the centres recipe
does not transfer unchanged, and why the two rows differ everywhere.

## Three things that were not obvious

**Loss is a poor progress signal.** An early centres run reached a *lower* loss
than the one that shipped, while being useless to search: a 20× slower target
refresh meant value never propagated outward from the solved state, so the
network fit a constant beautifully. Always check `solve.py`'s J-vs-depth table
instead, a collapsed network shows a flat line there, and nothing in the loss
curve reveals it.

**The wings need a curriculum.** Only states near solved have a grounded target:
those are the ones whose successors include the solved state, where the target is
a real 1 rather than a guess. Sampling `k ~ U(1,40)` over 63 moves puts ~2.5% of
the batch there, the squared error is dominated by the rest, and the run
collapsed to a constant J = 15.33 at every depth. Growing `k` only once the
network has fitted the current depth fixes it. Expect the loss to *rise* as `k`
does, that is the problem getting harder, not the training failing.

**fp16 for the target network is free speed.** It is ~1.6× faster and only ever
produces a regression target, never a gradient. The training forward/backward
stays fp32.

## Exporting to the app

```bash
../.venv-mlx/bin/python export_js.py     # -> src/solver/centers-net.js
```

The JS port runs the same beam search in the browser. It exploits the one-hot
encoding (the first matrix multiply only ever touches 24 of 144 inputs) because
a beam step cannot afford a dense 800k-parameter pass per state. That is also why
the shipped centres network is small: capacity that helps in Python is a latency
bill in the browser.

## What this cannot give you

A learned heuristic is not admissible: nothing stops the network
overestimating, so solutions found with it are near-optimal and never *proven*
minimal. That is the same trade DeepCubeA makes and it is unavoidable for a
learned value function. `src/solver/pdb.js` builds a genuinely admissible bound
by exhaustive search, and it bottoms out at 9 moves against a ~137-move solve;
see Figure 8.
