# What the corpus defects cost the published numbers

Measured 2026-09-10, before any restructuring, so that a number which moves later
can be attributed to a fixed defect rather than to a refactor. Reproduce with
`mlx-solver/defect_cost.py`; every figure below is written by it into
`eval/results/defect-cost.json`.

## The answer

**No number in the paper requires correction.** Both defects corrupt quantities
that never reach it.

**But the script producing the paper's figure cannot run.** That is a separate
problem and it is worse than a stale number, because nobody reproducing this work
can regenerate or check the figure at all.

## The headline figure is untouched

The tie-inflation figure quoted in the abstract, and again twice in the body, is
identical to six decimal places:

| | n | mean | max |
|---|---|---|---|
| committed baseline (`ac3a246`) | 8 | +0.0843 | +0.1012 |
| recomputed now | 8 | +0.0843 | +0.1012 |
| difference | | +0.000000 | |

It cannot have been affected, and the reason is structural rather than lucky. The
tie-inflation analysis reads no Profile corpus at all: it builds its own sample
from exact distance tables, over three hardcoded cube configurations. Neither the
missing deduplication nor the missing domain split has any path to it.

This contradicts what was assumed when the ticket was written, which assumed the
figure came from the loop with no deduplication. It does not. The two analyses sit
in the same file and one calls the other, which is what made the assumption
plausible, but they share no data.

The `0.101` quoted as the maximum is likewise unchanged.

## The deduplication defect is severe, and reaches nothing published

| | n | PDB n | PDB r | all r |
|---|---|---|---|---|
| as committed, no dedup | 125 | 73 | **+0.320** | +0.925 |
| with dedup applied | 54 | 21 | **+0.021** | +0.874 |

The correlation between predicted and measured pooled tau for pattern databases
collapses from +0.320 to +0.021, which is to say from a visible relationship to
none at all. The inflated count was manufacturing it: 73 rows standing for 21
distinct heuristics, each duplicate carrying the same values and so tightening the
apparent fit.

Learned heuristics are untouched at n=26 and r=+0.979, which is the expected
shape. A Profile is written once per learned checkpoint and re-measures the same
Rungs and the same random control on the same states, so the repeats fall on the
classical side and the random control, never on the learned one.

None of these numbers appears in the paper.

## The domain-pooling defect moves the spread-growth figure

| | PDB | learned |
|---|---|---|
| committed baseline, before the sliding tile existed | 0.149 (n=11) | 2.043 (n=21) |
| pooled across domains, as the code now does | 0.149 (n=11) | **2.626** (n=25) |
| cube only, the correction | 0.149 (n=11) | **2.043** (n=21) |
| sliding tile only | none pass the filter | **32.541** (n=4) |

Four sliding-tile Profiles move the learned figure by 29%, because the tile's
spread-growth is an order of magnitude above the cube's. Pattern databases are
unaffected: no sliding-tile Profile survives the four-Shell minimum on that side.

None of these numbers appears in the paper either.

Worth noting for later, not acted on here: 32.5 against 2.04 is the same "the two
are not the same thing" argument the analysis already makes between classes,
appearing again between domains. It is a finding, not a defect, and it belongs to
whichever ticket reports the split.

## The reproducibility failure

`tau_theory.py` raises a `TypeError` and produces nothing. Its tie-inflation
analysis imports `project` from `profiles`, and that function was removed in
`ad86483` when both domains were put behind one interface. The signature that
replaced it takes the task as its first argument.

The script has been unrunnable for every commit since, nine of them. And
`tau-theory.json` on disk was last written at `ac3a246`, which predates the
sliding-tile work entirely, so the file the paper's figure traces to is older than
the break.

`REPRODUCE.md` opens by promising that every number in the paper comes from a
command in it. For this figure that promise does not currently hold. The repair is
one call, and `defect_cost.py` demonstrates the figure reproduces exactly once it
is made.

## Consequences for the tickets that follow

- Ticket 03 migrates the corpus loops and will move the prediction correlations
  and the spread-growth figure. Both are internal; neither needs a paper edit.
- Ticket 05, wiring these numbers through the generated-table pipeline, is worth
  doing on its own merit rather than as a correction. The figure is right; it is
  the inability to regenerate it that is wrong.
- The one-call repair should land before or with ticket 03, since nothing in this
  file can be verified while it raises on import of its own dependency.
