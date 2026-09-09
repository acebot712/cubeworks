# CUBEWORKS

A 4x4 Rubik's-cube AR solver, and the heuristic-measurement study built on its
solver. These are the terms that have been used here to mean incompatible things.

## Language

### Measuring a heuristic

**Shell**:
The set of states at exact distance `d` from the goal.
_Avoid_: level, layer, ring

**Profile**:
A heuristic's adjacent-shell ordering accuracy as a function of `d`.
_Avoid_: curve, decay curve, per-shell accuracy

**GDRC**:
One Kendall tau-b between a heuristic and exact distance, pooled over every pair
of states. A tau stratified by distance is a Profile, not a GDRC. Tau-b and not
tau-a, because the tie handling is what the tau-b bias result is about.
_Avoid_: tau, rank correlation, pooled tau

**Decay**:
A Profile's first measured shell accuracy minus its last. Endpoints are the
shells that survived the minimum-population filter, so two Decays are comparable
only when measured over the same shells.
_Avoid_: drop, drop-off, degradation

**Strength**:
A heuristic's overall ordering quality, measured by GDRC.
_Avoid_: quality, accuracy, power

**d-prime**:
The between-shell gap in mean heuristic value divided by the within-shell spread.
_Avoid_: signal-to-noise, separation, effect size

### Abstractions

**Rung**:
A sub-problem tracking only some pieces, treating the rest as interchangeable.
_Avoid_: abstraction level, PDB level, sub-problem

**Degenerate rung**:
A rung that still separates every state of the full task, so its table is the
exact distance rather than an estimate. An oracle, not a baseline.
_Avoid_: perfect rung, exact rung

### Search

**Search cost**:
Expansions to goal under greedy best-first search with a closed list.
_Avoid_: nodes expanded, cost, budget, effort

**Required width**:
The smallest beam width reaching a target solve rate, measured by
`budget_curve.py`. A different quantity from Search cost.
_Avoid_: w90, beam budget, cost

**Instance**:
One start state a search is asked to solve.
_Avoid_: problem, case, sample, state

### Training runs

**Final checkpoint**:
The end-of-training checkpoint of one run. Two final checkpoints of the same
configuration differ only by seed and count as independent observations.
_Avoid_: model, net, run

**Intermediate snapshot**:
A checkpoint written partway through a run, named `@step`. Not independent of
the Final checkpoint of the run it came from, so the two never enter the same
test as separate observations.
_Avoid_: checkpoint, early model
