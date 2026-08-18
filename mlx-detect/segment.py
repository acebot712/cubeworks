"""Turn a rough box into an exact quadrilateral, using SAM.

The VLM is a good *finder* and a poor *measurer*: measured on real frames it
located the cube in 4/4 but its boxes sat at IoU 0.31-0.57 against the actual
face. Brute-forcing the classical search inside those boxes did not close the
gap either, the limit is precision, not search effort.

SAM is the opposite instrument. Prompted with a box it returns a pixel-exact
mask, and a mask has something a box does not: a boundary. Fitting a quad to
that boundary gives the four corners the homography actually needs.

Neither model is trained on cubes and neither needs a dataset.

Falls back cleanly: if torch or the weights are unavailable, `available()` is
False and the service keeps returning boxes as before.
"""
import numpy as np

_state = {"model": None, "processor": None, "device": None, "error": None}

# vit-base, not vit-huge. Measured: huge gave IDENTICAL results on every real
# frame (0.75/0.69/0.29/0.76 vs 0.75/0.70/0.29/0.76) for 2.7x the latency
# (4s -> 10.8s). Mask boundary quality was never the bottleneck: SAM-base
# already returns masks at 0.95-1.00 confidence. The limit is WHAT it segments
# (the whole cube, not the face), and a bigger encoder does not change that.
# Switch with:  npm run mlx -- --sam facebook/sam-vit-huge
MODEL = "facebook/sam-vit-base"


def available():
    return _state["model"] is not None


def load(name=MODEL):
    try:
        import torch
        from transformers import SamModel, SamProcessor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _state["processor"] = SamProcessor.from_pretrained(name)
        _state["model"] = SamModel.from_pretrained(name).to(device).eval()
        _state["device"] = device
        return True
    except Exception as e:  # missing torch, no weights, no network
        _state["error"] = f"{type(e).__name__}: {e}"
        return False


def _largest_component(mask):
    """Keep only the biggest blob: SAM occasionally returns specks alongside
    the object, and a stray speck would drag a corner across the frame."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    for sy in range(0, h, 2):
        for sx in range(0, w, 2):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if best is None or len(pts) > len(best):
                best = pts
    if not best:
        return None
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = zip(*best)
    out[np.array(ys), np.array(xs)] = True
    return out


def _convex_hull(points):
    """Andrew's monotone chain."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out
    return half(pts)[:-1] + half(reversed(pts))[:-1]


def quad_from_mask(mask):
    """Largest-area quadrilateral inscribed in the mask's convex hull.

    A cube face photographed at an angle is a general quadrilateral, so we do
    NOT fit a rotated rectangle, that would reintroduce the very assumption
    the perspective work removed.
    """
    m = _largest_component(mask)
    if m is None:
        return None
    ys, xs = np.nonzero(m)
    if len(xs) < 32:
        return None
    hull = _convex_hull(np.stack([xs, ys], axis=1))
    if len(hull) < 4:
        return None
    # decimate so the O(n^4) search stays cheap on a ragged hull
    if len(hull) > 40:
        step = len(hull) / 40.0
        hull = [hull[int(i * step)] for i in range(40)]

    n = len(hull)
    tri = lambda a, b, c: abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    ) / 2.0
    best, best_area = None, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    a = tri(hull[i], hull[j], hull[k]) + tri(hull[i], hull[k], hull[l])
                    if a > best_area:
                        best_area, best = a, (hull[i], hull[j], hull[k], hull[l])
    if best is None:
        return None

    # order as TL, TR, BR, BL so the homography is not mirrored or rotated
    pts = list(best)
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    import math
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))   # CCW from +x
    start = min(range(4), key=lambda i: pts[i][0] + pts[i][1])  # top-left-most
    pts = pts[start:] + pts[:start]
    return [[float(x), float(y)] for x, y in pts]


def refine(image, box_norm):
    """image: PIL RGB. box_norm: [x0,y0,x1,y1] in 0..1.
    -> {"corners": [[x,y] x4] normalised, "mask_area": float} or None"""
    if not available():
        return None
    import torch

    w, h = image.size
    box = [box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h]
    proc, model = _state["processor"], _state["model"]
    inputs = proc(image, input_boxes=[[box]], return_tensors="pt")
    # Metal has no float64, and the processor emits the box prompt as one.
    inputs = {
        k: (v.to(torch.float32) if torch.is_floating_point(v) else v).to(_state["device"])
        if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]                                   # (num_masks, H, W)
    scores = out.iou_scores.cpu().numpy().reshape(-1)
    order = np.argsort(-scores)

    # Return EVERY plausible quad rather than guessing which one is the face.
    # SAM's highest-scoring mask is usually the whole 3D cube, its silhouette
    # includes the side faces, which is why that quad scores badly as a "face".
    # SAM proposes; the caller's cube-face scorer disposes. Each component is
    # then used for what it is actually good at.
    out = []
    for idx in order:
        m = masks[idx].numpy().astype(bool)
        quad = quad_from_mask(m)
        if quad is None:
            continue
        sides = [
            np.hypot(quad[(i + 1) % 4][0] - quad[i][0], quad[(i + 1) % 4][1] - quad[i][1])
            for i in range(4)
        ]
        if min(sides) < 1e-3 or max(sides) / min(sides) > 2.6:
            continue
        out.append({
            "corners": [[x / w, y / h] for x, y in quad],
            "mask_area": float(m.sum()) / (w * h),
            "sam_score": float(scores[idx]),
        })
    if not out:
        return None
    # smallest first: on a cube held at an angle the FACE is a sub-part of the
    # silhouette, so the tighter masks are the more promising candidates
    out.sort(key=lambda c: c["mask_area"])
    return {"candidates": out, **out[0]}
