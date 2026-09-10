# Reproducing every result

Every number in the paper comes from a JSON file in `eval/results/`, and every
one of those files is produced by a command below. Nothing is transcribed by
hand. If a command is re-run, the file it writes is overwritten and the figure
that reads it moves.

Two things are stated up front because a reader should not have to discover them:
which results are **superseded**, and which measurements we **could not obtain**.

---

## 0. Environment

Measured on the machine that produced every number in the paper:

| | |
| --- | --- |
| hardware | Apple M5 Pro, 48 GB unified memory |
| OS | macOS (Darwin 25.4) |
| Python | 3.14.6 |
| MLX | 0.32.0 (`mlx-metal` 0.32.0) |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |

Exact pins in `mlx-solver/requirements-lock.txt`. MLX requires Apple silicon;
the training scripts will not run on other hardware without substitution of the
array backend. Everything in `exact.py` and `probe_exact.py` is pure NumPy and
is portable.

```bash
python3 -m venv .venv-mlx
./.venv-mlx/bin/pip install -r mlx-solver/requirements-lock.txt
```

---

## 1. One command for everything

```bash
cd mlx-solver && ./run_pipeline.sh
```

Idempotent: each step skips work whose output already exists, and training
resumes from checkpoints written every 4,000 steps. Kill it, close the lid, cut
the power: re-running loses at most a few minutes of the step in flight. A PID
lock prevents two copies racing and clears itself if a previous run was killed.

Progress at any time, without any of this tooling running:

```bash
cd mlx-solver && ./status.sh        # snapshot;  -w to refresh every 15s
```

Total cost from nothing: **~32 GPU-hours**, dominated by the ladder sweep.

---

## 2. Individual experiments

### Exact ground truth (`exact.py`)

Exhaustive BFS over a whole rung. Feasible to k=6; the k=6 table is 191 MB and
takes ~22 minutes.

```bash
for k in 2 3 4 5 6; do
  ../.venv-mlx/bin/python exact.py --k $k --save-table
done
# depth variants, same state space, different diameter
for ms in oi-q oi-q4 oi-q3; do
  for k in 2 3 4; do
    ../.venv-mlx/bin/python exact.py --k $k --moves $ms --save-table
  done
done
```

Writes `eval/results/exact-k*.json` (tracked) and `mlx-solver/exact_k*.npy`
(gitignored; 191 MB at k=6, regenerable).

**Self-check:** the script aborts if the reachable-set size disagrees with
P(24,k). A mismatch is a bug, not a result.

### E1: the size ladder (`sweep.py`)

8 rungs × 3 seeds × 3 budgets = 72 measurements. **~29 hours.**

```bash
../.venv-mlx/bin/python sweep.py --steps 40000 --seeds 3 --n 200 --width 100
```

Each run trains once to 40,000 steps and snapshots at 3,000 and 12,000, so the
extra budget points cost nothing. → `eval/results/ladder.json`

### E2: depth versus size (`depth_vs_size.py`)

The confound-breaking experiment: 4 generating sets on rung k=4, all reaching
exactly 255,024 states, at diameters 6/8/10/12. **~2.2 hours.**

```bash
../.venv-mlx/bin/python depth_vs_size.py --steps 12000 --seeds 3 --n 200 --width 100
```

→ `eval/results/depth-vs-size.json`

### E3/E4: per-shell profiles (`profiles.py`)

Learned heuristic vs genuine pattern databases (the exact table of a smaller
rung, admissible by abstraction) vs a random control, with pooled GDRC reported
alongside the per-shell profile. Seconds per config.

Omitting `--pdb-k` reports **every** abstraction available on the rung, which is
the point: with a single weak abstraction the comparison is uninterpretable, and
an earlier version of this study drew a false conclusion from exactly that.

```bash
for ms in all oi-q oi-q4 oi-q3; do for s in 0 1 2; do
  ../.venv-mlx/bin/python profiles.py --task wings-k4 --tag "_dv-${ms}_s${s}"
done; done
for s in 0 1 2; do
  ../.venv-mlx/bin/python profiles.py --task wings-k4 --tag "_s${s}"
  ../.venv-mlx/bin/python profiles.py --task wings-k6 --tag "_s${s}"
done
```

→ `eval/results/profile-*.json`

Each Profile records the sampler's actual arguments, not just its name and the
state count, because `n` is a product: 112500 is 2500x45 and also 4500x25, and
those are different distributions over shells. An analysis that needs the full
distance histogram redraws the sample from those arguments.

To check that every Profile still describes the sample it was measured on, and
to recover the arguments for one written before they were recorded:

```bash
../.venv-mlx/bin/python profiles.py --record-sampler-args          # dry run
../.venv-mlx/bin/python profiles.py --record-sampler-args --write
```

It redraws and compares against the shell sizes the Profile itself recorded, and
refuses to write anything it cannot reproduce. That is what makes a recovered
argument trustworthy: a wrong pair would have to land on twenty independent shell
counts by coincidence. `profiles.py --selftest` checks it actually rejects one.

### E4b: is the decay about training, or about strength? (`strength_control.py`)

Reads every profile above and asks whether learned heuristics decay more than
abstractions of the same strength. Deduplicates: a profile is written per learned
checkpoint and re-measures the same abstractions on the same states, so those
repeats are collapsed (without this, n reads 98 instead of 39 and every p-value
is far too small). Seconds.

```bash
../.venv-mlx/bin/python strength_control.py
```

→ `eval/results/strength-control.json`, and `paper/tables/strength.tex`,
`tables/matched.tex`, figure 14.

### Ground truth beyond enumeration (`probe_exact.py`, `profile_probed.py`)

Meet-in-the-middle exact distances for a sample, on rungs too large to enumerate.

**Verify before trusting it.** The prober must agree with exhaustive BFS
wherever both can run:

```bash
../.venv-mlx/bin/python probe_exact.py --k 4 --forward 3 --back 4 --verify 200
../.venv-mlx/bin/python probe_exact.py --k 6 --forward 4 --back 5 --verify 300
```

Both return exact agreement (200/200 and 300/300). Then:

```bash
../.venv-mlx/bin/python profile_probed.py --task wings-k8 --tag _s0 \
    --forward 6 --back 4 --cap 100000000 --per-len 60 --max-len 14
```

→ `eval/results/probeprofile-wings-k8_s0.json`

### Solve quality on both domains (`evaluate.py`, `resolution.py`)

Solve rate, saturation and, where an exact table exists, the optimality gap.
Both take a task of either domain and ask it where its table lives, so a board
reads a board's table:

```bash
../.venv-mlx/bin/python evaluate.py --task wings-k4 --tag _s0 --n 200 --width 100
../.venv-mlx/bin/python evaluate.py --task tile-3x3 --tag _s1 --n 200 --width 100
../.venv-mlx/bin/python resolution.py --task tile-3x3 --tag _s1
```

→ `eval/results/eval-<task><tag>.json`, `eval/results/resolution-<task><tag>.json`

The sliding-tile checkpoints solve 200/200 and are optimal on 98-99% of solves,
worst excess +1. Those numbers were unobtainable until the table path stopped
being composed by hand: a board's rung number lives in the cube's rung
namespace, so `tile-3x3` asked for `exact_k8.npy`, a cube rung's table. It is
absent, the optimality block was skipped, and nothing said so. A rung with no
table now names the table and the command that builds it, in the console and in
the JSON's `optimality_unavailable`.

### E6: the two-moment law (`dprime_law.py`)

Tests whether per-shell ordering accuracy is `Phi(gap / (sd * sqrt(2)))`, with no
fitted parameters, where `gap` is the between-shell mean difference and `sd` the
within-shell spread. Reads the profiles above; holds out the k=8 probed profile
and the loss-comparison checkpoints, whose ground truth and training objective
respectively come from elsewhere. Seconds.

Two things this script learned the hard way, both worth keeping if you adapt it:

- **Split the sample.** `profiles.py` records moments estimated on one half of
  each shell alongside accuracy measured on the other. Computing both from the
  same states manufactures agreement. Run that version on a heuristic with *no
  signal at all* and it reports a correlation of +0.95 where the truth is 0. The
  split figures are the ones any claim should rest on.
- **De-duplicate.** A profile file is written per learned checkpoint and
  re-measures the same abstractions on the same states, so those rows repeat
  verbatim. Counting them read n = 615 where the truth is 252, and gave pattern
  databases four times their weight. Same identity key as `strength_control.py`.
- **Report the noise floor.** Per-shell accuracy comes from 20,000 sampled pairs,
  so it carries ~0.002 of binomial error. Without that figure there is no way to
  tell a mean absolute error of 0.004 from a perfect fit measured imprecisely.

```bash
../.venv-mlx/bin/python dprime_law.py
```

→ `eval/results/dprime-law.json`

### E6b: break the law on purpose (`noise_dose.py`)

Everything in E6 is observational. This injects calibrated per-state noise into a
trained heuristic (`h(s) + lambda * eps(s)`, with `eps` derived by hashing the
state's index so the result is still a function of the state) and asks whether
the prediction survives. Rung, network and training are identical across the
sweep by construction, so nothing is confounded; and the noise family is chosen
to include one, Cauchy, that has no finite variance and that the law therefore
has no right to fit. Minutes, no training.

```bash
../.venv-mlx/bin/python noise_dose.py
```

→ `eval/results/noise-dose.json`

The error tracks tail weight monotonically: uniform 0.0042, Gaussian 0.0046,
Laplace 0.0123, **Cauchy 0.1654**, so the normal form, not mere monotonicity in
gap/sd, is carrying the prediction. The `forward` column is the stronger test: it
predicts perturbed accuracy from the *unperturbed* moments plus the analytic
variance inflation, touching the perturbed heuristic only to score it.

### E7: pooled tau, derived from the profile (`tau_theory.py`)

Three things, all checked against measurement rather than asserted:

- pooled tau-b written exactly as an `n_s * n_t`-weighted average of per-shell
  ordering accuracy over every pair of shells;
- the tie bias, a tie-free heuristic that orders every shell perfectly is capped
  at 0.919 on the k=6 sample, checked against scipy, while the integer-valued exact distance scores
  1.000. Each pattern database collects **+0.07 to +0.10 of pooled tau from ties
  alone**, which is why comparing a continuous learned heuristic against an
  integer abstraction by GDRC is not a like-for-like comparison;
- decay is exactly zero when within-shell spread is constant across shells, so
  the decay exists only because spread grows with distance.

```bash
../.venv-mlx/bin/python tau_theory.py
```

→ `eval/results/tau-theory.json`

### E8: does targeting the right quantity help? (`p3_losses.py`)

Three training objectives, identical in every other respect: `l2` (standard
DAVI), `rank` (RankNet-style pairwise logistic), and `dprime` (l2 plus a
within-shell variance penalty). **Hours.** Idempotent, re-run to resume.

All arms use `--gate order`. The default curriculum gate advances on squared
error, which a scale-free ranking loss can never satisfy; using it would pin
that arm at the first curriculum level and manufacture a result.

```bash
../.venv-mlx/bin/python p3_losses.py --steps 12000 --seeds 3
# the penalty-weight sweep, written to its OWN file: p3_losses.py rewrites its
# output wholesale, so sharing one would replace the comparison with the sweep
../.venv-mlx/bin/python p3_losses.py --steps 12000 --seeds 3 \
    --configs wings-k6 --arms "dprime:3,dprime:10,dprime:30" --out p3-lambda.json
```

→ `eval/results/p3-losses.json`, `eval/results/p3-lambda.json`

**Outcome: negative, and worth reading before building on the law.** Nothing beat
squared error. The variance penalty is inert at lambda=1 and monotonically harmful
above it (Spearman between penalty weight and decay = +1.00), because suppressing
within-shell spread makes outputs more constant, the curriculum gate reads that as
failure to separate adjacent shells, and training stalls, at lambda=10 the run
never leaves scramble depth 7 of 40. Record `k_cur` for every run; without it this
looks like the objective failing rather than the curriculum stalling.

### E5: does the profile predict search cost? (`e5_predict.py`)

Pairs each config's intrinsic profile against the smallest beam width reaching a
target solve rate, then correlates. **~1 hour.**

```bash
../.venv-mlx/bin/python e5_predict.py --n-budget 100
```

→ `eval/results/budget-*.json`, `eval/results/e5-prediction.json`

---

## 3. Figures

```bash
npm run dev                       # then open http://localhost:5183/figures.html
```

Every figure renders from `eval/results/*.json`. The footer button writes all of
them to `eval/figures/` as standalone SVG. Two schematics (Figures 1 and 2)
illustrate a mechanism rather than report a measurement and are labelled as such
on the page.

To rasterise on macOS without extra tooling:

```bash
qlmanage -t -s 1400 -o png eval/figures/figure-13.svg
```

---

## 4. Superseded results, kept rather than deleted

`eval/results/superseded-walk-sampler/` holds 19 profile files produced with the
**earlier sampling scheme**, which drew states from deep random walks. That
concentrates almost every state into the two or three shells nearest the mean
distance, so the shallow half of each profile was measured on a handful of states
or not at all. They are superseded by the `sweep` sampler, which draws across
scramble lengths 1..16 and then *measures* true distance rather than assuming the
scramble length is the distance.

This is not a cosmetic change. Under the old sampler, E5 reported pooled GDRC at
Pearson −0.841 against the per-shell measure's −0.923, a modest gap. Under the
corrected sampler the same comparison is **−0.391 against −0.809**. A
methodological fix moved a headline number substantially, which is a reason to
distrust the original and a reason the old files are kept where they can be
inspected.

### The single-abstraction profile comparison

Profiles were originally run with `--pdb-k 2`, comparing the learned heuristic
against one abstraction. On that basis the paper claimed learned heuristics decay
with distance *while pattern databases do not*. **That claim is withdrawn.**
PDB(k=2) scores a pooled τ of only +0.570 on rung k=6; it is flat because it is
close to chance, not because abstraction confers distance-invariance. With every
abstraction reported, the steepest decay on that rung belongs to PDB(k=5), the
strongest heuristic in the comparison, and once strength is controlled the two
classes are statistically indistinguishable (`strength_control.py`).

The old single-baseline JSONs are not preserved separately because re-running
`profiles.py` without `--pdb-k` regenerates a strict superset: the k=2 column is
still present in every current profile file, alongside the abstractions whose
absence caused the error.

---

## 5. Measurements we could not obtain

Stated because their absence shapes what the paper can claim.

**Ground truth stops at k=8.** Exhaustive enumeration is infeasible past k=6
(96,909,120 states at k=6; the next rung is 300× larger). Meet-in-the-middle
buys exactly one further rung. A k=10 run was attempted and abandoned after
**31 hours**: its mean distance is ≈9.9 (from the fitted `0.864k + 1.242`) while
a radius-6 ball with 4 backward levels reaches only distance 10, so roughly half
its states would have returned unresolved, the deep half the profile exists to
measure. Covering it needs a ball larger than memory allows. **This is the same
wall that stops enumeration, arriving one rung later**, and it is why every rung
above k=8 is reported by solve rate alone.

**E5's spread is thin.** 11 of 19 configs reach 90% solve at beam width 1, so the
correlation rests on 8 points that vary. It is reported as evidence consistent
with the claim, not as the claim established.

**Depth and branching are not separated.** The depth experiment fixes state-space
cardinality exactly and varies the generating set, but fewer generators also mean
a smaller branching factor. The design rules **size** out; it cannot apportion the
remainder between depth and branching.

---

## 6. Seeds and determinism

`--seed` sets both the NumPy scramble stream and MLX weight initialisation, so a
run is reproducible given identical library versions. Evaluation scrambles use a
separate fixed seed (1234) held constant across every configuration, so two
checkpoints are always compared on the same states.

Run-to-run variation across seeds is a reported result, not noise to average
away: it peaks in the transition region (±23.5 percentage points at k=10) and is
near zero where the method clearly works or clearly fails. One seed of the
deepest depth-variant collapses entirely while its siblings succeed. Single-seed
results would have supported the wrong conclusion in both places.

---

## 7. What is tracked and what is not

Tracked, small, and the source of every figure:

- `eval/results/*.json`
- `eval/figures/*.svg`
- `paper/related-work.md`
- all scripts, `requirements-lock.txt`

Gitignored, large and regenerable:

- `mlx-solver/ckpt_*.npz`, checkpoints, 155 MB each (network plus Adam moments)
- `mlx-solver/exact_k*.npy`, distance tables, 191 MB at k=6
- `mlx-solver/*.log`, `eval/figures/png/`
