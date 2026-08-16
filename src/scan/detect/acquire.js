// Optional model-assisted acquisition.
//
// The classical cold search has a measured squeeze: people hold the cube at
// 10-25% of frame height, small faces need a ~640px-wide comb search to resolve
// their seam period, and that search costs 75-100ms. Tracking a face that is
// already located costs 3.2ms and does not care about resolution.
//
// So only ACQUISITION is expensive, and that is the one step a pretrained model
// does in a single forward pass. This module asks a local MLX service for a
// rough box; the caller then runs the ordinary corner refinement INSIDE that
// box, which is both far cheaper than a full-frame scan and far more accurate
// than the model's box on its own.
//
// The service is optional. If it is not running, callers fall back to the
// classical search — nothing here is required for the app to work.

const DEFAULT_URL = 'http://127.0.0.1:8765';

let available = null;   // null = unknown, true/false once probed
let probing = null;

export function acquireUrl() {
  try {
    const q = new URLSearchParams(window.location.search).get('mlx');
    if (q) return q === '1' ? DEFAULT_URL : q;
  } catch { /* not in a browser */ }
  return DEFAULT_URL;
}

// Cheap one-shot probe so the app can show whether the assist is live.
export function probeService(url = acquireUrl()) {
  if (probing) return probing;
  probing = fetch(`${url}/health`, { signal: AbortSignal.timeout(1500) })
    .then((r) => r.json())
    .then((h) => { available = !!h.ok; return h; })
    .catch(() => { available = false; return null; })
    .finally(() => { probing = null; });
  return probing;
}

export function serviceAvailable() { return available; }

// frame: {data, width, height} RGBA.
// -> array of PROPOSAL boxes in frame pixels, best-guess first, or [].
//
// Deliberately plural. Measured on real frames: the VLM's box and each of SAM's
// masks are all plausible, and which one is actually the FACE varies per frame —
// SAM's top-scoring mask is usually the whole 3D cube, whose silhouette includes
// the side faces. So the service proposes and the caller's cube-face scorer
// disposes, which beat every single-proposal pipeline tried (3/4 vs 2/4).
export async function acquireFace(frame, { url = acquireUrl(), timeoutMs = 20000 } = {}) {
  if (available === false) return [];
  const jpeg = await frameToJpeg(frame);
  if (!jpeg) return [];
  try {
    const r = await fetch(`${url}/detect`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ image: jpeg }),
      signal: AbortSignal.timeout(timeoutMs),
    }).then((x) => x.json());
    available = true;
    const out = [];
    const toBox = (corners) => {
      const xs = corners.map((p) => p[0] * frame.width);
      const ys = corners.map((p) => p[1] * frame.height);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    };
    // SAM's masks first: they are tighter than the VLM box and won on every
    // real frame measured
    for (const c of r.candidates || []) {
      if (c.corners) out.push({ box: toBox(c.corners), from: 'sam' });
    }
    if (r.box) {
      out.push({
        box: [r.box[0] * frame.width, r.box[1] * frame.height, r.box[2] * frame.width, r.box[3] * frame.height],
        from: 'vlm',
      });
    }
    return out;
  } catch {
    available = false;
    return [];
  }
}

function frameToJpeg(frame) {
  if (typeof document === 'undefined') return Promise.resolve(null);
  const c = document.createElement('canvas');
  c.width = frame.width; c.height = frame.height;
  const ctx = c.getContext('2d');
  ctx.putImageData(new ImageData(new Uint8ClampedArray(frame.data), frame.width, frame.height), 0, 0);
  return Promise.resolve(c.toDataURL('image/jpeg', 0.85));
}
