# What the corpus defects cost the published numbers

Measured 2026-09-10, before any restructuring, so that a number which moves later
can be attributed to a fixed defect rather than to a refactor.

**This is a frozen record, not a regenerable one.** The figures below were
produced by `mlx-solver/defect_cost.py`, a measurement harness written to hold
the defects and their corrections side by side. Doing that meant carrying a fifth
copy of the corpus loop with deduplication switchable, which is exactly what the
tickets below existed to remove, so the script said in its own docstring that it
should be deleted once ticket 03 had landed and its numbers were recorded. It has,
they are, and it is: deleted once tickets 01 through 08 had all landed and the
restructuring it was measuring was finished. `git log -- mlx-solver/defect_cost.py`
has it, if the harness itself is ever wanted again.

What survives is this document and `eval/results/defect-cost.json`, which holds
every figure below at full precision. Neither can be regenerated, because the
defects they measure no longer exist. What CAN be re-run is the thing that
matters, the published figure: see the last section.

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

## The reproducibility failure, and its repair

At the time of measurement `tau_theory.py` raised a `TypeError` and produced
nothing. Its tie-inflation analysis imported `project` from `profiles`, and that
function was removed in `ad86483` when both domains were put behind one
interface. The signature that replaced it takes the task as its first argument.

The script had been unrunnable for every commit since, nine of them. And
`tau-theory.json` on disk was last written at `ac3a246`, which predates the
sliding-tile work entirely, so the file the paper's figure traced to was older
than the break.

`REPRODUCE.md` opens by promising that every number in the paper comes from a
command in it. For this figure that promise did not hold. **It holds now.** The
repair landed with ticket 03 in `ba0d68f`, and the figure has been re-derived
through every change since:

```bash
cd mlx-solver && ../.venv-mlx/bin/python tau_theory.py
```

→ `eval/results/tau-theory.json`, whose `tie_inflation` block gives mean
**+0.0843** and max **+0.1012** over its 8 rows: the same values as the `ac3a246`
baseline and as the recomputation recorded here. That is the check this document
was written to make possible, and it is now a live command rather than a claim.

Fixing the import also surfaced a second latent fault in the same file, a `vc`
name bound inside `check_prediction` and referenced in `main`, wrong since the
initial commit and unreachable while the first error fired first.

## What became of the tickets that followed

All eight landed. Against the predictions made here:

- **Ticket 03** (`ba0d68f`) migrated the corpus loops and moved the prediction
  correlations and the spread-growth figure, as expected. Both internal; no paper
  edit.
- **Ticket 05** (`5f36b1d`) wired the numbers through the generated-table
  pipeline. It also found what this document did not: a *different* stale
  published number, the 0.917 and 0.600 gaps, caused by the same domain-pooling
  defect in a third file. So "no number in the paper requires correction" was true
  of the two defects measured here and not of the family they belong to.
- **The one-call repair** landed with ticket 03 as recommended.
- **Tickets 06 and 07** (`141e7a7`, `9098e54`) moved the domain behaviour onto the
  tasks and deleted the dispatch functions, which is what removes the conditions
  for all of these defects rather than the defects themselves.
- **Ticket 08** (`5f78f7a`) found the same shape once more in the exact-table path,
  where a board asked for a cube rung's table and lost its optimality gap in
  silence.
