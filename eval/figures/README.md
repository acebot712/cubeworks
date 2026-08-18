# Figures

Nine figures covering the perception and search stack, at
<http://localhost:5183/figures.html> while the dev server is running.

Every plot is rendered from a results file written by an actual run. Two
schematics (Figures 1 and 2) illustrate a mechanism rather than report a
measurement, and are labelled as such on the page. Nothing is typed in by hand, a chart built from remembered numbers is a drawing, and once rendered there is no
way to tell the two apart.

## Watching a training run

The page polls `/__training` every 15s and shows a live panel: progress bar,
ETA, curriculum depth, and sparklines for J spread and loss. Figures 3 and 4
track the same run, so a job in progress needs no manual refresh: `npm run
metrics` is only needed to freeze the numbers into a checked-in file.

The status dot is green while the trainer is still writing, amber once it has
gone quiet for more than ~2.5 checkpoint intervals, blue when it reaches its
target step. Terminal equivalent:

```bash
npm run watch:train
```

## Regenerating

```bash
npm run bench          # solver measurements -> eval/results/*.json
npm run bench:pdb      # ...plus the 51.5M-state pattern-database BFS (~13 min)
npm run metrics        # training logs -> eval/results/training-*.json
```

Then reload the page. `npm run metrics` is safe to run while training is still
going, it picks up whatever has been written so far, so Figures 3 and 4 track a
run in progress.

## Exporting

Each figure has an **SVG** button. The footer button writes all nine to this
directory at once. They are standalone SVG with the fonts named rather than
embedded, so they scale cleanly and stay editable.

To rasterise on macOS without extra tooling:

```bash
qlmanage -t -s 1400 -o png figure-3.svg
```

## What each one shows

| # | figure | source |
| - | ------ | ------ |
| 1 | System overview | schematic |
| 2 | How DAVI works, and why it needs a curriculum | schematic |
| 3 | The value function taking shape across checkpoints | `training-wings.json` |
| 4 | Training loss against curriculum depth | `training-wings.json` |
| 5 | Learned centres search vs the hand-written solver | `centres-greedy-vs-learned.json` |
| 6 | What beam width buys, and what it costs | `centres-beam-width.json` |
| 7 | Where the moves go, per phase | `full-solve.json` |
| 8 | Pattern database: an exact bound that is too weak | `pdb-depths.json` |
| 9 | Why the 4×4 is not the 3×3 | published state counts |

Figure 3 is the one to look at first when judging a training run. Figure 4's
loss curve will happily fall while the network is collapsing.
