# MLX acquisition service

A local, Apple-silicon service that answers one question: **roughly where is the
cube face in this frame?**

## Why only that

The classical detector was measured on real captured frames and has a structural
squeeze:

| | cost | works? |
| --- | --- | --- |
| cold search at 640px wide | 75–100 ms | finds small faces |
| cold search at 256px wide | ~20 ms | **misses them** (score 0.673 → 0.197) |
| tracking an already-found face | **3.2 ms** | resolution-independent |

People hold a cube at 10–25% of frame height. A small face's seam period is only
~4px at 256, so the comb search cannot resolve it, and an exhaustive comb at
640 is roughly 8× too slow to run live.

Only *acquisition* is expensive, and acquisition is exactly what a pretrained
model does in one forward pass. So the model finds the cube once; the existing
corner refinement then runs **inside its box** and produces the exact quad, and
the 3.2 ms tracker takes over from there.

That split matters: the model's known weakness is imprecise coordinates, and
that weakness is irrelevant here because it never has to be precise. Everything
that must be exact: corners, rectification, colour reading, cube validation: stays deterministic and unit-tested.

## Running

```bash
npm run mlx                       # ./.venv-mlx/bin/python mlx-detect/server.py
npm run mlx -- --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit   # smaller/faster
```

First run downloads the model (~5 GB for the 7B 4-bit) to `~/.cache/huggingface`.
The service listens on `http://127.0.0.1:8765`.

The app auto-detects it. If the service is not running, the app falls back to
the classical search: nothing here is required.

## Measuring

```bash
npm run eval:mlx            # model box recall + IoU on the captured frames
npm run eval                # the classical detector, same frames
```

Both read `eval/labels.json`, so the two are directly comparable. The model eval
scores *box* IoU (0.5 pass), because the model only needs to get close enough
for the corner descent to finish the job.

Calibration, measured against a stub that returns the labelled box inflated by
10% on each side: roughly what a good model box looks like: **IoU ≈ 0.69**. So
0.5 is a lenient bar by design; it fails a box that is in the wrong place, not
one that is merely imprecise. Judge the model on *box recall* first and IoU
second, since imprecision is what the ROI refinement exists to absorb.

## API

```
GET  /health   -> {"ok": true, "model": "...", "loaded": true}
POST /detect   {"image": "data:image/jpeg;base64,..."}
               -> {"found": true, "box": [x0,y0,x1,y1], "norm": true, "ms": 812}
```

Boxes are normalised to 0–1, so the caller can send whatever resolution it likes.

## What the pipeline does, and why

Two pretrained models, neither trained on cubes, neither needing a dataset:

1. **Qwen2.5-VL** finds the cube. Measured on real frames it located it in 4/4,
   with 0 false positives, but its boxes sat at IoU 0.31-0.57 against the
   actual face. It is a good finder and a poor measurer.
2. **SAM** measures. Prompted with that box it returns pixel-exact masks, and a
   mask has a boundary a box does not. Fitting a quadrilateral to that boundary
   gives four corners.
3. **The classical cube-face scorer decides.** SAM's top-scoring mask is usually
   the whole 3D *cube*, whose silhouette includes the side faces, a bad "face"
   even though it is a good mask. So the service returns EVERY plausible quad
   and the caller scores them all, refines each with the corner descent, and
   keeps the winner.

Measured on real captured frames, score above the 0.50 acquire threshold:

| pipeline | frames found |
| --- | --- |
| classical alone | 2 / 4 |
| VLM box → ROI search | 3 / 4 |
| **VLM + all SAM masks → scorer picks** | **3 / 4**, and higher score on *every* frame |

Per frame, best-of vs classical: 0.75/0.72, 0.70/0.65, 0.29/0.19, 0.76/0.38.

Things that did **not** work, recorded so they are not retried:

- **SAM-huge instead of SAM-base.** Identical results on every real frame
  (0.75/0.69/0.29/0.76 vs 0.75/0.70/0.29/0.76) for 2.7x the latency, 4s → 10.8s.
  Mask *boundary quality* was never the bottleneck: SAM-base already returns
  masks at 0.95-1.00 confidence. The limit is *what* it segments: the whole 3D
  cube rather than the face, and a bigger encoder does not change that. Still
  available via `--sam facebook/sam-vit-huge` if you want to re-check.

- Asking the VLM for "the face" rather than "the cube": moved one frame from
  IoU 0.47 to 0.52 and nothing else.
- Dense multi-start search inside the ROI: actively *worse* (0.72 → 0.42). The
  comb-seeded descent already starts in better places than a grid does. More
  compute was never the missing ingredient.

## Model choice

Default is `Qwen2.5-VL-7B-Instruct-4bit`: Qwen2.5-VL is one of the stronger open
models at *grounding* (returning real coordinates rather than just describing the
image), and 4-bit at 7B is comfortable on 48 GB. Swap to the 3B if acquisition
latency matters more than hit rate; the eval script is the way to decide.
