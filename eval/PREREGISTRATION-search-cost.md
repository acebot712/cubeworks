# Pre-registration: does the profile predict search cost better than pooled GDRC?

Written 2026-09-09, before any search in this experiment has been run. Committed
first so that the git timestamp is the evidence. If anything below changes after
the first run, it goes in the Deviations section at the bottom, with a reason,
rather than being edited away.

## Why this is pre-registered at all

An earlier version of this work claimed the stratified measure predicts required
beam width where pooled GDRC does not. It did not survive its own censoring: 11
of 19 configurations reached the target at width 1, and on the 8 where the
outcome varied the ordering reversed in favour of pooled GDRC. The section was
removed rather than reported.

Two things went wrong there, and only one of them was censoring. The predictor,
accuracy near the diameter, was chosen after seeing data. With enough summaries
of a profile to choose from, something will separate, and nothing about the
result would then be believable. So the predictor, the outcome, the test and the
threshold are all fixed here in advance.

This project has had five claims refuted by its own controls. A pre-registered
null is worth more than a sixth positive that a referee inverts.

## The question

Does an intrinsic, search-free property of a heuristic predict the search it
needs, and specifically does the adjacent-shell profile predict it BETTER than
Wilt and Ruml's pooled GDRC?

Not whether a ranking statistic predicts search cost. Wilt and Ruml established
that, and it is not ours. The only part that could be ours is whether measuring
adjacent-shell pairs beats measuring all pairs.

## Prediction

The profile predicts search cost better than pooled GDRC. Stated as a direction
before running: the difference of Spearman correlations is positive.

## Design, fixed in advance

**Search.** Greedy best-first search, counting expansions. Chosen over beam
search because at fixed beam width the expansion count is `width x branching x
levels`, which is solution length in disguise and has almost no dynamic range.
Greedy best-first is also what Wilt and Ruml measured, so the comparison against
the prior work is like for like.

**Duplicate detection.** A closed list in both domains, implemented as a boolean
array indexed by `domains.rank`, which is a perfect hash into `[0, cells)`. Exact,
no collisions. Identical treatment in both domains, because an asymmetry here
would sit directly on the axis being measured.

**No censoring.** Greedy best-first with a closed list on a connected state space
always reaches the goal, and cannot expand more than `|S|` nodes. Every task
below is small enough that this bound is affordable, so cost is defined for every
(heuristic, instance) pair and no censored-data estimator is needed. This is why
`wings-k5` and `wings-k6` are out of scope: at 5.1M and 96.9M states the bound
stops being affordable, and including them would reintroduce the censoring that
the previous attempt died of.

**Budget confound, acknowledged.** `budget_curve.py` argues that any quantity
measured at a fixed budget is a property of (heuristic x algorithm x budget) and
not of the heuristic, which is why this project previously used required width.
That argument is correct and does not apply here, because the claim is
comparative rather than absolute: every heuristic meets the identical algorithm
and the identical bound, so the confound is constant across the comparison and
cannot produce the effect. The paper must say this explicitly; a reader who
followed the earlier reasoning will otherwise think it was forgotten.

**Cap by expansions, never by wall clock.** A time cap would systematically favour
pattern databases, an array lookup, over learned heuristics, a forward pass, and
would manufacture exactly the learned-versus-classical difference this work spent
five controls ruling out.

**Tasks.** Six, each analysed separately.

| task | states | branching | rungs |
|---|---|---|---|
| tile-2x4 | 20,160 | 4 | 1 to 4 |
| tile-3x3 | 181,440 | 4 | 1 to 5 |
| wings-k4, moves `all` | 255,024 | 63 | 2 to 3 |
| wings-k4, moves `oi-q` | 255,024 | reduced | 2 to 3 |
| wings-k4, moves `oi-q3` | 255,024 | reduced | 2 to 3 |
| wings-k4, moves `oi-q4` | 255,024 | reduced | 2 to 3 |

The restricted movesets are different state graphs with their own exact tables,
so they are different tasks, not different heuristics on one task.

**Instances.** 1000 uniform random reachable states per task, one fixed set,
identical across every heuristic on that task, so every comparison is paired and
instance difficulty is held constant rather than averaged over. Deliberately not
the swept sample `profiles.py` uses: that sample is non-uniform by design, and
reusing it would correlate the predictor with the outcome through the sampling.

**Population.** Every available pattern-database rung, every final learned
checkpoint, per task. Roughly 80 observations across the six tasks. Intermediate
`@step` checkpoints are excluded because they are not independent of their final
versions.

**Predictor spread, and its cost.** No synthetic heuristics. A calibrated noise
ladder was considered and rejected: it would have swept a base heuristic down a
continuous strength range with everything else held fixed, but injected noise
moves the within-shell spread directly, and that spread is the term the predictor
is built from. A correlation manufactured that way would test whether cost
follows a quantity we had just moved by hand, not whether the profile predicts
anything.

The price is paid on the cube. `wings-k4` offers two rungs plus a tight cluster
of learned checkpoints, so roughly three or four distinct strength levels per
task, which is thin for estimating a correlation. The `oi-q3` task is better
placed because its `p3` loss variants degrade quality monotonically with penalty
weight and so span a real range on their own. The sliding tile carries most of
the spread, with five rungs from near chance to near oracle on `tile-3x3`.

This is stated in advance rather than discovered afterwards: if the result is
null, thin cube spread is a live explanation for it, and the per-task correlations
required by the analysis below will show which tasks had the range to detect
anything.

**Random control.** Excluded from the primary regression, reported separately as
an anchor showing cost responds at all. A chance-level heuristic sits at one
extreme corner of every scatter and would inflate both correlations; it is a
control that the measurement works, not a member of the population being
described.

## The analysis, fixed in advance

**Predictor.** Mean adjacent-shell ordering accuracy, weighted by shell size. One
summary, fixed. Note the contrast this creates: pooled GDRC is a shell-size
weighted sum over ALL pairs, and the predictor is the same weighting over
ADJACENT pairs only. Adjacent-only versus all-pairs is the dilution argument, and
it is the only contrast that is not tautological.

**Outcome.** Median expansions to goal over the 1000 instances. Geometric mean is
a declared robustness check. Under a rank-based test the two will rarely differ,
which is exactly why the choice is pinned rather than made afterwards.

**Test.** Within each task, Spearman(predictor, cost) minus Spearman(GDRC, cost).
The difference is the statistic, not either correlation alone: both will be high
simply because both track heuristic strength. Uncertainty by paired bootstrap
resampling heuristics within task, 10,000 resamples, 95 percent interval.
Combined across the six tasks by a fixed-effects average.

Spearman is rank based, so the heavy tail in expansion counts needs no transform.

**Success.** Both of: the combined difference's 95 percent interval excludes
zero, and the difference is positive in at least 4 of the 6 tasks. The
majority-of-tasks clause exists so that one task cannot carry the result, which
is how the previous attempt failed.

**Failure.** Anything else, including a positive combined interval carried by two
tasks.

## Secondary outcomes, subordinate and declared now

1. Solution suboptimality: returned path length minus exact optimal distance,
   which is free because the exact tables are already built.
2. Geometric mean as the outcome, as a robustness check.

None of these may be promoted to the headline if the primary comes out null.
They are written here so that promotion is visible if it happens.

## What gets written if the result is null

The null, at the same prominence a positive would get, in the section that
currently reads "we do not show that any of this makes a search cheaper". That
sentence becomes a measured answer instead of an admission, which is an
improvement whichever way the number falls.

Explicitly not permitted: trying a different profile summary until something
separates. That is the forking path that produced the removed section.

## What this will not claim

- That a ranking statistic predicting search cost is novel. Wilt and Ruml (2016).
- That search cost is independent of state-space size or solution depth.
  Korf, Reid and Edelkamp contradict it, and the design holds branching and depth
  constant within a task rather than varying them.
- That a result on these six tasks extends to classical planning.
- That expansions under greedy best-first predict cost under any other algorithm.

## Deviations from this pre-registration

None yet. Anything that changes after the first run is recorded here, dated, with
the reason.
