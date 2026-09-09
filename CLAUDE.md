# CUBEWORKS

Vocabulary: [CONTEXT.md](CONTEXT.md). Commands, pinned versions, superseded
results and measurements that could not be obtained: [REPRODUCE.md](REPRODUCE.md).

## Running anything

`.venv-mlx/bin/python`, never `python3`. The system interpreter has no numpy, so
every script here dies on import.

## Changing a number

Scripts write to `eval/results/*.json`; `paper/build_tables.py` reads those and
writes `paper/tables/*.tex`. Re-run the measurement, then re-run
`build_tables.py`. `paper/make_abstract.py` derives `abstract.txt` from
`main.tex` and enforces arXiv's 1920-character cap.

## Prose

Where an em dash would go, use a comma, a colon or a full stop. This holds in
code comments, commit messages, markdown and the paper alike.

## Two domains, one code path

Both state spaces go through `mlx-solver/domains.py`, and a measurement script
takes whichever task it is handed without branching on domain. Forking a script
per domain would let a difference in shell binning or pair sampling surface as a
domain effect, which is the one thing the second domain exists to detect.

## Deduplicate before pooling

A profile is written per learned checkpoint and re-measures the same rungs on the
same states, so those rows repeat verbatim. This project has made that mistake in
three separate scripts; once it inflated n from 252 to 615 and gave pattern
databases four times their true weight. The identity key is `(task, moves, name)`,
plus the tag when the heuristic is learned.

Its sibling: a rung that still separates every state is an oracle rather than a
baseline, and `domains.rungs` drops it.

## eval/frames

Photographs of a person and a home interior. They stay out of the repository and
out of every archive, permanently. Excluding only the `.jpg` achieves nothing,
because the `.bin` are the same images as raw RGB. See `eval/README.md`.
