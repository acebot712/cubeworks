# Real-frame evaluation set

The detector's unit tests are synthetic — images generated from the same
assumptions the detector encodes. That is circular, and it stayed green through
every real-world failure so far (curtains read as a cube, light-bodied cubes
invisible, non-standard palettes unresolvable). This directory is the fix: real
frames, from a real camera, in real rooms.

## Capturing

```bash
npm run dev
```

Open <http://localhost:5183/capture.html> (a dev tool, deliberately separate
from the app). Point the camera at a cube, click its four corners in order —
top-left, top-right, bottom-right, bottom-left — then **Save**. The dashed teal
box shows what the detector currently thinks, so disagreements are visible while
you label.

| key | action |
| --- | --- |
| `space` | hold the frame so a moving cube can be labelled |
| `backspace` | undo the last corner |
| `enter` | save with corners |
| `n` | save as a negative (no cube in frame) |

**Negatives matter as much as positives.** Two of the three real failures were
false positives on background. Bank plenty of frames of the places it has
false-fired: curtains, bookshelves, bare skin, keyboards, empty desks.

Fill in the **conditions** field (`white-body, lamp-lit, tilted`) — the report
breaks results down by tag, which is how you find out *which* conditions fail
rather than just that something does.

Aim for variety over volume: different cubes, rooms, lighting, distances,
angles. A few hundred well-spread samples beats a thousand of the same shot.

## Running

```bash
npm run eval                    # full report
npm run eval -- --verbose       # list every failure
npm run eval -- --threshold 0.4 # sweep the acquire threshold
npm run eval -- --iou 0.75      # demand tighter localisation
```

Exits non-zero if recall drops below 90% or the false-positive rate exceeds 2%,
so it can gate a commit.

Failures are listed by id — open `eval/frames/<id>.jpg` to see exactly what the
detector saw.

## Format

`labels.json` is a list of samples:

```jsonc
{
  "id": "0001",
  "corners": [[0.42, 0.31], ...],  // normalised [0,1], or null for a negative
  "tags": "white-body, lamp-lit",
  "detW": 192, "detH": 108,        // detection-frame size
  "videoW": 1280, "videoH": 720,
  "predicted": { "cx": .., "score": .. },  // what it said at capture time
  "at": "2026-07-29T..."
}
```

Each sample has two files in `frames/`:

- `<id>.jpg` — full-resolution frame, for human inspection and for training a
  model later.
- `<id>.bin` — the exact downscaled RGB buffer the detector consumes
  (`detW × detH × 3`, row-major). The eval runner reads this directly, so it
  needs no image decoder and no dependencies.

`frames/` is gitignored and is **deliberately absent from the public repository**,
including from any release archive. The captures are photographs of a person and
a home interior, and note that excluding only the `.jpg` would achieve nothing:
`<id>.bin` is the same picture as raw `detW × detH × 3` RGB and reconstructs in a
few lines of numpy. Both are out, permanently.

Nothing else depends on them — no result in the paper reads `frames/`, and the
rest of the test suite passes without it. To run this eval, capture your own set
with `capture.html`; `labels.json` is tracked so the format and the label schema
are still there to follow.
