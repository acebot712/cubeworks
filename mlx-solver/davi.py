"""Deep Approximate Value Iteration with the operational bits that matter:
a live ETA, checkpoint/resume, and a metrics log that the figure script reads.

Runs either sub-problem:

    --task centers   24 centre facelets, 3.25e15 states   (trains in minutes)
    --task wings     24 edge wings,      3.10e23 states   (~95 million x larger)

Resume is the point of the checkpoints. A wings run is long enough that losing
it to a crash, a closed lid, or an accidental Ctrl-C would be genuinely costly,
so state is written every --ckpt-every steps and picked up automatically:

    ../.venv-mlx/bin/python davi.py --task wings --steps 200000
    ../.venv-mlx/bin/python davi.py --task wings --steps 200000   # resumes

Method (DeepCubeA's, not AlphaZero's, a cube is single-agent shortest path,
so there is no adversarial self-play):

    J(s) = 0                             if solved
    J(s) = min over moves of 1 + J_target(s')   otherwise

Targets come from a periodically-frozen copy of the network. That refresh rate
matters more than it looks: at one point a centres net trained with a 20x
slower refresh reached a LOWER loss while being useless to search, because the
value function never propagated outward from the solved state. Loss is not the
metric: `--task X --probe` measures whether J actually tracks distance.
"""
import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = Path(__file__).parent


# --------------------------------------------------------------------------
DONT_CARE = 24          # the symbol for "this slot holds a piece we are not tracking"


def rung_size(k):
    """Reachable states on wing rung k: place k distinguishable pieces in 24 slots.

    P(24, k) = 24! / (24-k)!.  At k = 24 the cube's parity constraint halves it: below that the untracked pieces absorb parity, so every arrangement is reachable.
    """
    n = math.factorial(24) // math.factorial(24 - k)
    return n // 2 if k >= 23 else n


class Task:
    """A sub-problem: how to permute it, what solved means, how to encode it.

    Three families:
      centers      the 24 centre facelets, 6 indistinguishable groups of 4
      wings        all 24 edge wings, every one distinguishable
      wings-k<N>   a LADDER RUNG: track N of the 24 wings, treat the rest as
                   interchangeable

    The rungs exist to answer one question: how does a learned heuristic degrade
    as the state space grows?, so everything except state-space size is held
    fixed across them. Same 63 moves, same 24 slots, same 600-wide encoding
    (24 slots x 25 symbols, the 25th being "not tracked"). Only N varies, and
    with it the reachable count, from 552 at N=2 to 3.10e23 at N=24.

    Tracking a subset is exactly the projection a pattern database uses, so each
    rung is a genuine sub-problem of the cube and not a synthetic toy: a move
    still acts deterministically on the record of where the tracked pieces are.
    """

    def __init__(self, name, moves="all"):
        self.name = name
        is_centers = name == "centers"
        tbl = json.loads((HERE / f"{'center' if is_centers else 'wing'}_perms.json").read_text())
        self.tokens = tbl["tokens"]
        self.perms = np.array(tbl["perms"], dtype=np.int32)
        # Restricting the generating set is how size and DEPTH are separated.
        # Dropping half-turns generates exactly the same group (X2 = X.X), so the
        # reachable state count is untouched while the diameter grows, which is
        # the only way to ask whether performance tracks how BIG a problem is or
        # how DEEP it is. On the unrestricted ladder those two are perfectly
        # correlated (diameter = k+2), so neither can be blamed for a failure.
        self.moveset = moves
        keep = self._select(moves)
        self.tokens = [self.tokens[i] for i in keep]
        self.perms = self.perms[keep]
        self.n_moves = len(self.tokens)
        self.n_slots = 24

        if is_centers:
            # the value in each slot is which FACE is showing
            self.k = None
            self.n_sym = 6
            self.solved = np.repeat(np.arange(6, dtype=np.int8), 4)
        elif name.startswith("wings-k"):
            self.k = int(name.split("wings-k")[1])
            if not 2 <= self.k <= 24:
                raise ValueError(f"rung k must be in 2..24, got {self.k}")
            # 25 symbols on every rung, so the input width never changes and the
            # only thing separating two rungs is how many pieces are real
            self.n_sym = DONT_CARE + 1
            self.solved = np.full(24, DONT_CARE, dtype=np.int8)
            self.solved[:self.k] = np.arange(self.k, dtype=np.int8)
        else:
            self.k = 24
            self.n_sym = 24
            self.solved = np.arange(24, dtype=np.int8)

        self.n_in = self.n_slots * self.n_sym
        self.size = rung_size(self.k) if self.k else \
            math.factorial(24) // math.factorial(4) ** 6

    # Generating sets that all reach the SAME states, verified by exhaustive BFS
    # at k=4 (255,024 every time), but at different diameters. This is the only
    # handle we have on depth that does not also change the state space, and it
    # is what separates "the problem is big" from "the problem is deep".
    #
    # Measured diameter at k=4:   all 6  |  oi-q 8  |  oi-q4 10  |  oi-q3 12
    #
    # The price is that fewer generators also means a smaller branching factor,
    # so depth and branching move together here. Size does not, and since size
    # is held exactly constant, any change in performance across these sets
    # rules size out as the cause, which is the claim that matters.
    # faces allowed, as single-letter bases (outer UPPER, inner slice lower)
    MOVESETS = {
        "all":     None,                 # every generator, including wides and rotations
        "quarter": "*",                  # everything except half turns
        "oi-q":    "UDRLFBudrlfb",
        "oi-q4":   "UDRLudrl",
        "oi-q3":   "UDRudr",
    }

    def _select(self, moves):
        """-> indices of the generators to keep."""
        if moves not in self.MOVESETS:
            raise ValueError(f"unknown move set '{moves}'; have {list(self.MOVESETS)}")
        spec = self.MOVESETS[moves]
        if spec is None:
            return list(range(len(self.tokens)))
        idx = []
        for i, t in enumerate(self.tokens):
            if t.endswith("2"):          # every restricted set is quarter-turn only
                continue
            base = t.rstrip("'")
            if spec == "*":
                idx.append(i)
            elif len(base) == 1 and base in spec:
                idx.append(i)
        if not idx:
            raise ValueError(f"move set '{moves}' selected nothing")
        return idx

    def apply(self, states, move_ids):
        out = np.empty_like(states)
        for m in range(self.n_moves):
            sel = move_ids == m
            if sel.any():
                out[np.ix_(sel, self.perms[m])] = states[sel]
        return out

    def children(self, states):
        B = states.shape[0]
        out = np.empty((B, self.n_moves, self.n_slots), dtype=np.int8)
        for m in range(self.n_moves):
            out[:, m, :][:, self.perms[m]] = states
        return out

    def scramble(self, batch, k_max, rng):
        states = np.tile(self.solved, (batch, 1))
        ks = rng.integers(1, k_max + 1, size=batch)
        for step in range(k_max):
            active = ks > step
            if not active.any():
                break
            states[active] = self.apply(states[active], rng.integers(0, self.n_moves, size=int(active.sum())))
        return states, ks

    def encode(self, states):
        B = states.shape[0]
        oh = np.zeros((B, self.n_in), dtype=np.float32)
        idx = np.arange(self.n_slots) * self.n_sym + states
        oh[np.arange(B)[:, None], idx] = 1.0
        return oh

    def is_solved(self, states):
        return (states == self.solved).all(axis=1)


def batch_order_acc(net, task, states, y, rng=np.random.default_rng(0)):
    """How often the net orders two states from adjacent target shells correctly.

    Scale-invariant, so it means the same thing for a regression loss and for a
    ranking loss whose outputs can sit at any magnitude. Used as the curriculum
    gate when the arms have to be compared against each other.
    """
    h = np.array(net(mx.array(task.encode(states))), copy=False)
    sh = np.rint(y).astype(np.int32)
    accs = []
    for v in np.unique(sh):
        a, b = h[sh == v], h[sh == v + 1]
        if a.size < 8 or b.size < 8:
            continue
        m = min(4096, a.size * b.size)
        ia, ib = rng.integers(0, a.size, m), rng.integers(0, b.size, m)
        accs.append(np.mean(a[ia] < b[ib]) + 0.5 * np.mean(a[ia] == b[ib]))
    return float(np.mean(accs)) if accs else 0.0


class ValueNet(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        dims = (n_in,) + tuple(hidden)
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.head = nn.Linear(hidden[-1], 1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.head(x).squeeze(-1)


# --------------------------------------------------------------------------
def cast_tree(tree, dtype):
    if isinstance(tree, dict):
        return {k: cast_tree(v, dtype) for k, v in tree.items()}
    if isinstance(tree, list):
        return [cast_tree(v, dtype) for v in tree]
    return tree.astype(dtype)


def save_ckpt(path, net, opt):
    flat = {f"net.{k}": np.array(v, copy=False) for k, v in tree_flatten(net.parameters())}
    flat.update({f"opt.{k}": np.array(v, copy=False) for k, v in tree_flatten(opt.state)
                 if isinstance(v, mx.array)})
    np.savez(path, **flat)


def read_ckpt(blob, prefix):
    pairs = [(k[len(prefix):], mx.array(blob[k])) for k in blob.files if k.startswith(prefix)]
    return tree_unflatten(pairs) if pairs else None


def fmt_eta(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


PROBE_DEPTHS = (1, 2, 3, 5, 8, 12, 16, 20, 26, 32)


def probe(task, net, rng, depths=PROBE_DEPTHS):
    """Does J rise with scramble depth? A flat line means the network collapsed,
    which a falling loss will happily hide."""
    out = []
    for k in depths:
        st = np.tile(task.solved, (256, 1))
        for _ in range(k):
            st = task.apply(st, rng.integers(0, task.n_moves, size=256))
        j = float(np.array(net(mx.array(task.encode(st))), copy=False).mean())
        out.append((k, j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="centers",
                    help="centers | wings | wings-k<N> for a cube ladder rung, "
                         "N in 2..24 | tile-RxC or tile-RxC-kN for a sliding tile")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the scramble stream AND the weight init, so a "
                         "sweep over seeds measures real run-to-run variance")
    ap.add_argument("--moves", default="all",
                    help="generating set: all | quarter | oi-q | oi-q4 | oi-q3. "
                         "All reach the same states; they differ only in diameter, "
                         "which is how depth is varied at constant problem size")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint/metric filenames, e.g. a seed label")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--kmax", type=int, default=0, help="0 = task default")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="")
    ap.add_argument("--sync-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--snapshot-at", default="",
                    help="comma list of steps to freeze a separate copy at, e.g. "
                         "3000,12000,40000. One training run then yields several "
                         "budget points, which is what turns 'does it fail' into "
                         "'how much compute does it need', and costs nothing extra")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--probe", action="store_true", help="probe a trained net and exit")
    ap.add_argument("--half", choices=["auto", "on", "off"], default="auto",
                    help="evaluate the frozen target net in fp16. It is ~1.6x faster "
                         "and only ever produces a regression target, never a gradient; "
                         "auto = on for wings, off for centres")
    ap.add_argument("--curriculum", choices=["auto", "on", "off"], default="auto",
                    help="grow the scramble depth instead of sampling U(1,kmax) from "
                         "the start; auto = on for wings, off for centres")
    ap.add_argument("--loss", choices=("l2", "rank", "rank-adj", "dprime"), default="l2",
                    help="l2 = standard DAVI regression; rank = RankNet-style pairwise "
                         "logistic over uniformly drawn pairs; rank-adj = the same loss "
                         "restricted to pairs one shell apart, which are the comparisons a "
                         "search actually makes. Uniform pairs are mostly far apart and "
                         "easy, so plain `rank` optimises exactly the diluted quantity "
                         "this work argues against and is not the fair form of the "
                         "comparator; dprime = l2 plus a within-shell variance penalty, "
                         "the term dprime_law.py identifies as governing ordering")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the within-shell variance penalty (--loss dprime)")
    ap.add_argument("--pairs", type=int, default=4096,
                    help="pairs sampled per step (--loss rank)")
    ap.add_argument("--gate", choices=("loss", "order"), default="loss",
                    help="curriculum advance criterion. 'loss' is the original and "
                         "keeps published runs reproducible; 'order' is scale-invariant "
                         "and is required whenever arms with different loss scales are "
                         "being compared to each other")
    ap.add_argument("--adv-order", type=float, default=0.94,
                    help="adjacent-shell ordering accuracy to advance at (--gate order). "
                         "0.94 is where a heuristic with the default --adv-loss of 0.20 "
                         "sits, by acc = Phi(1/(sqrt(0.20)*sqrt(2)))")
    ap.add_argument("--adv-loss", type=float, default=0.20,
                    help="advance the curriculum once the running loss falls below this")
    ap.add_argument("--adv-min-steps", type=int, default=300,
                    help="minimum steps to spend at each curriculum level")
    args = ap.parse_args()

    # Local import: domains imports this module for Task, so a module-level
    # import here would be circular. The indirection buys the sliding-tile
    # domain without davi.py having to know anything about it.
    from domains import make_task
    task = make_task(args.task, moves=args.moves)
    is_wing = args.task != "centers"
    kmax = args.kmax or (24 if not is_wing else 40)
    # Every rung gets the SAME architecture. Sizing each one to its own state
    # space would confound the very thing we are measuring, we would no longer
    # know whether a rung failed because it was large or because its net was small.
    hidden = tuple(int(x) for x in args.hidden.split(",")) if args.hidden \
        else ((256, 128) if not is_wing else (4096, 2048, 1024))

    stem = f"{args.task}{args.tag}"
    ckpt = HERE / f"ckpt_{stem}.npz"
    meta_path = HERE / f"ckpt_{stem}.json"
    metrics_path = HERE / f"metrics_{stem}.jsonl"

    half = args.half == "on" or (args.half == "auto" and is_wing)
    tdtype = mx.float16 if half else mx.float32

    # Why a curriculum. Bootstrapping only has an anchor where a batch contains
    # states near solved: those are the ones whose successors include the solved
    # state, so their target is a real 1 rather than a guess. Sampling k~U(1,40)
    # over 63 moves puts ~2.5% of the batch there, and the squared-error loss is
    # dominated by the 97.5% whose targets are noise, a wings run set up that
    # way collapsed to a constant J = 15.33 at every depth. Growing k only once
    # the network has fitted the current depth keeps the anchor in view.
    curric = args.curriculum == "on" or (args.curriculum == "auto" and is_wing)
    k_cur = 2 if curric else kmax

    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)
    net = ValueNet(task.n_in, hidden)
    target = ValueNet(task.n_in, hidden)
    mx.eval(net.parameters(), target.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    start_step = 0
    if ckpt.exists() and not args.fresh:
        blob = np.load(ckpt)
        net.update(read_ckpt(blob, "net.") or {})
        mx.eval(net.parameters())
        # Adam's moments are part of the training state. Resuming without them
        # restarts the optimiser cold and visibly dents the loss for a few
        # thousand steps, which on a long run is worth avoiding.
        moments = read_ckpt(blob, "opt.")
        if moments is not None:
            opt.state = moments
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            start_step = saved.get("step", 0)
            k_cur = saved.get("k_cur", k_cur)
        print(f"resumed from {ckpt.name} at step {start_step:,}  k={k_cur}"
              f"{'' if moments is not None else '  (no optimiser moments in checkpoint)'}",
              flush=True)

    def sync_target():
        target.update(cast_tree(net.parameters(), tdtype))
        mx.eval(target.parameters())

    sync_target()

    if args.probe:
        print(f"{args.task}: does J track distance?")
        for k, j in probe(task, net, rng):
            print(f"   k={k:3d}   J={j:7.2f}")
        return

    n_params = sum(v.size for _, v in tree_flatten(net.parameters()))
    print(f"task {args.task}  |  {task.n_in} inputs  |  hidden {hidden}  |  "
          f"{n_params:,} params  |  {task.n_moves} moves  |  k~U(1,{kmax})")
    print(f"target steps {args.steps:,}  batch {args.batch}  "
          f"target net {'fp16' if half else 'fp32'}  "
          f"curriculum {'k=2 -> ' + str(kmax) if curric else 'off'}  "
          f"checkpoint every {args.ckpt_every:,}\n", flush=True)

    # Three objectives, identical in every other respect. See dprime_law.py: a
    # heuristic's per-shell ordering accuracy is Phi(gap / (sd * sqrt(2))),
    # where gap is the between-shell mean difference and sd the within-shell
    # spread. Adjacent shells are one move apart, so gap is pinned near 1 for
    # anything calibrated to distance; the only variable left to move is sd.
    #
    #   l2      the standard DAVI objective. Penalises bias and variance
    #           identically, because it is a regression loss and does not know
    #           that only ordering matters.
    #   rank    RankNet-style pairwise logistic on states drawn from the same
    #           batch: the "optimise to rank, not to estimate" comparator.
    #   dprime  l2 plus an explicit penalty on within-shell variance -- the one
    #           term the law says governs ordering. Shells are formed from the
    #           bootstrap target, which is the only distance estimate available
    #           at training time.
    UNUSED = mx.zeros((1,), dtype=mx.int32)

    def l2_loss(model, x, y, gm, gmn, pi, pj):
        return mx.mean(mx.square(model(x) - y))

    def rank_loss(model, x, y, gm, gmn, pi, pj):
        h = model(x)
        # pi/pj index pairs with y[pi] < y[pj], so h[pj] - h[pi] should be
        # positive; log(1 + exp(-z)) written stably
        z = h[pj] - h[pi]
        return mx.mean(mx.logaddexp(mx.zeros_like(z), -z))

    def dprime_loss(model, x, y, gm, gmn, pi, pj):
        h = model(x)
        # gmn has column-normalised indicator columns, so h @ gmn is the vector
        # of per-shell means; gm is the raw indicator, which scatters each mean
        # back to its members. The residual is then h minus its own shell's
        # mean -- within-shell spread, differentiably.
        shell_mean = mx.matmul(h[None, :], gmn).squeeze(0)
        resid = h - mx.matmul(gm, shell_mean[:, None]).squeeze(-1)
        return mx.mean(mx.square(h - y)) + args.lam * mx.mean(mx.square(resid))

    loss_fn = {"l2": l2_loss, "rank": rank_loss, "rank-adj": rank_loss,
               "dprime": dprime_loss}[args.loss]
    grad_fn = nn.value_and_grad(net, loss_fn)
    t0 = time.time()
    running = None
    done = 0

    snapshots = {int(s) for s in args.snapshot_at.split(",") if s.strip()}

    level_start = start_step
    for step in range(start_step + 1, args.steps + 1):
        states, _ = task.scramble(args.batch, k_cur, rng)
        kids = task.children(states).reshape(-1, task.n_slots)
        jt = np.array(target(mx.array(task.encode(kids), dtype=tdtype)).astype(mx.float32),
                      copy=False).reshape(args.batch, task.n_moves)
        jt = np.maximum(jt, 0.0)
        jt[task.is_solved(kids).reshape(args.batch, task.n_moves)] = 0.0
        y = (1.0 + jt).min(axis=1)
        y[task.is_solved(states)] = 0.0

        # Shells and pairs for the two non-L2 objectives. The bootstrap target
        # is the only distance estimate available during training, so shells are
        # formed by rounding it; states whose target is unique in the batch
        # contribute no within-shell variance and drop out harmlessly.
        gm = gmn = pi = pj = UNUSED
        if args.loss == "dprime":
            _, idx = np.unique(np.rint(y).astype(np.int32), return_inverse=True)
            n_sh = int(idx.max()) + 1
            oh = np.zeros((args.batch, n_sh), dtype=np.float32)
            oh[np.arange(args.batch), idx] = 1.0
            gm = mx.array(oh)
            gmn = mx.array(oh / np.maximum(oh.sum(axis=0, keepdims=True), 1.0))
        elif args.loss in ("rank", "rank-adj"):
            a = rng.integers(0, args.batch, size=args.pairs)
            b = rng.integers(0, args.batch, size=args.pairs)
            if args.loss == "rank-adj":
                # only pairs one shell apart: the comparison a search makes, and
                # the one a uniform sample almost never draws
                keep = np.abs(np.rint(y[a]) - np.rint(y[b])) == 1
                if not keep.any():          # early steps have one shell only
                    keep = y[a] != y[b]
            else:
                keep = y[a] != y[b]
            a, b = a[keep], b[keep]
            if a.size == 0:
                continue
            lo = np.where(y[a] < y[b], a, b)
            hi = np.where(y[a] < y[b], b, a)
            pi, pj = mx.array(lo), mx.array(hi)

        loss, grads = grad_fn(net, mx.array(task.encode(states)),
                              mx.array(y.astype(np.float32)), gm, gmn, pi, pj)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        done += 1

        lv = float(loss)
        running = lv if running is None else 0.98 * running + 0.02 * lv

        if step % args.sync_every == 0:
            sync_target()
            # What counts as "this level is learned" has to mean the same thing
            # in every arm. Squared error does not: a rank loss is scale-free,
            # so its outputs drift to whatever magnitude they like and its L2
            # never falls, which would pin it at the first level forever. The
            # order gate measures adjacent-shell ordering accuracy instead --
            # scale-invariant, and the quantity the experiment is about.
            if args.gate == "order":
                gate_ok = batch_order_acc(net, task, states, y) >= args.adv_order
            else:
                gate_ok = running < args.adv_loss
            if (curric and k_cur < kmax and gate_ok
                    and step - level_start >= args.adv_min_steps):
                k_cur += 1
                level_start = step
                # the new depth is unfitted by construction, so restart the
                # average here rather than coasting on the old level's value
                running = lv

        if step in snapshots:
            snap = HERE / f"ckpt_{stem}@{step}"
            save_ckpt(snap.with_suffix(".npz"), net, opt)
            snap.with_suffix(".json").write_text(json.dumps(
                {"step": step, "task": args.task, "k_cur": k_cur,
                 "hidden": list(hidden), "kmax": kmax, "moves": args.moves,
                 "snapshot": True}))
            print(f"  snapshot @ {step:,}", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(ckpt, net, opt)
            meta_path.write_text(json.dumps({"step": step, "task": args.task, "k_cur": k_cur,
                                             "hidden": list(hidden), "kmax": kmax,
                                             "moves": args.moves}))
            # a full J-vs-depth curve, not just endpoints, this is the figure
            # that shows whether the value function is actually taking shape
            pr = probe(task, net, rng,
                       depths=sorted({*(d for d in PROBE_DEPTHS if d <= kmax), kmax}))
            with metrics_path.open("a") as f:
                f.write(json.dumps({"step": step, "loss": running, "k": k_cur,
                                    "probe": pr, "elapsed": time.time() - t0}) + "\n")
            rate = done / (time.time() - t0)
            eta = (args.steps - step) / max(rate, 1e-9)
            spread = pr[-1][1] - pr[0][1]     # J(deep) - J(shallow): must stay > 0
            print(f"  step {step:7,}/{args.steps:,}  k {k_cur:2d}  loss {running:8.4f}  "
                  f"J spread {spread:6.2f}  {rate:5.1f} it/s  ETA {fmt_eta(eta)}", flush=True)

    print(f"\ndone in {fmt_eta(time.time() - t0)}  ->  {ckpt.name}")


if __name__ == "__main__":
    main()
