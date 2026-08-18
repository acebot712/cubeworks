// Coarse localization by comb filtering, the "finder pattern" step.
//
// A cube face is periodic structure: 5 equally spaced edges per axis (2 borders
// + 3 seams). Rather than hunt a peaky 4-D score with a lattice, we project edge
// energy onto each axis and exhaustively comb-search the 1-D profiles for that
// period, the same trick QR finders use on scanlines, and immune to the local
// maxima that trap a window search.
import { DEG, downsample } from './frame.js';
import { scoreWindow } from './score.js';

const THETA_SWEEP = [-16, -8, 0, 8, 16]; // degrees
// Real hand-held frames put the face at 10-25% of frame height; the old 0.28
// floor meant the search never even looked at the size people actually use.
const MIN_FACE_H = 0.09;  // face height as a fraction of the frame
const MAX_FACE_H = 0.92;
const REL_FLOOR = 0.60;   // comb strength relative to the profile's own energy
const SQUARENESS = 0.3;   // allowed mismatch between the two axes' periods
const MAX_SEEDS = 3;
// Profiles cost O(W*H) per rotation, so the coarse pass runs on a downscaled
// copy and its seeds are scaled back up. Only the fine descent needs the full
// frame, and that samples a fixed number of points regardless of size.
const COARSE_W = 640;

// Edge energy projected onto the two axes of a theta-rotated frame.
function gradProfiles(frame, theta) {
  const { data, width: W, height: H } = frame;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  // profile axes are rotated; pad generously so all pixels land in range
  const span = Math.ceil(W * Math.abs(cos) + H * Math.abs(sin)) + 2;
  const spanV = Math.ceil(W * Math.abs(sin) + H * Math.abs(cos)) + 2;
  const pu = new Float32Array(span), pv = new Float32Array(spanV);
  const offU = Math.ceil(H * Math.abs(sin)), offV = Math.ceil(W * Math.abs(sin));
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = (y * W + x) * 4;
      const l = data[i] + data[i + 1] + data[i + 2];
      const lx = data[i + 4] + data[i + 5] + data[i + 6];
      const ly = data[i + W * 4] + data[i + W * 4 + 1] + data[i + W * 4 + 2];
      const gx = Math.abs(lx - l) / 3, gy = Math.abs(ly - l) / 3;
      const u = Math.round(x * cos + y * sin) + offU;
      const v = Math.round(-x * sin + y * cos) + offV;
      if (u >= 0 && u < span) pu[u] += gx;
      if (v >= 0 && v < spanV) pv[v] += gy;
    }
  }
  return { pu, pv, offU, offV };
}

// Best 5-tooth comb (o, o+p, .., o+4p) minus the anti-comb at cell centres.
function combSearch(prof, pMin, pMax) {
  const L = prof.length;
  const at = (i) => {
    const k = Math.round(i);
    if (k < 0 || k >= L) return 0;
    // tolerate 1px seam wander
    let m = prof[k];
    if (k > 0 && prof[k - 1] > m) m = prof[k - 1];
    if (k + 1 < L && prof[k + 1] > m) m = prof[k + 1];
    return m;
  };
  let best = null;
  for (let p = pMin; p <= pMax; p += 0.5) {
    const maxO = L - 1 - 4 * p;
    for (let o = 0; o <= maxO; o += 1) {
      let on = 0, off = 0;
      for (let k = 0; k <= 4; k++) on += at(o + k * p);
      for (let k = 0; k < 4; k++) off += at(o + (k + 0.5) * p);
      const s = on / 5 - 0.85 * (off / 4);
      if (!best || s > best.s) best = { s, o, p };
    }
  }
  if (best) {
    // strength relative to the profile's own energy, an absolute score means
    // nothing across scenes, but "this comb beats the average edge" does
    let sum = 0, nn = 0;
    for (let i = 0; i < L; i++) if (prof[i] > 0) { sum += prof[i]; nn++; }
    const mean = nn ? sum / nn : 0;
    best.rel = mean > 0 ? best.s / mean : 0;
  }
  return best;
}

// -> up to MAX_SEEDS candidate poses, best-scoring first
export function coarseSeeds(frame) {
  const small = downsample(frame, COARSE_W);
  const up = frame.width / small.width;   // scale seeds back to full resolution
  const pMin = Math.max(3, (MIN_FACE_H * small.height) / 4);
  const pMax = (MAX_FACE_H * small.height) / 4;
  const seeds = [];

  for (const deg of THETA_SWEEP) {
    const theta = deg * DEG;
    const { pu, pv, offU, offV } = gradProfiles(small, theta);
    const bu = combSearch(pu, pMin, pMax);
    const bv = combSearch(pv, pMin, pMax);
    if (!bu || !bv) continue;
    // both axes must show comb structure: a curtain scores ~2.1 on the fold axis
    // and ~0.2 on the other, while the weakest true face measures ~1.0 on both
    if (Math.min(bu.rel, bv.rel) < REL_FLOOR) continue;
    // the face is square: reject wildly mismatched periods
    if (Math.abs(bu.p - bv.p) / Math.max(bu.p, bv.p) > SQUARENESS) continue;

    // rotate the (u,v) centre back into image coords
    const cos = Math.cos(theta), sin = Math.sin(theta);
    const cu = bu.o + 2 * bu.p - offU;
    const cv = bv.o + 2 * bv.p - offV;
    const cx = (cu * cos - cv * sin) * up;
    const cy = (cu * sin + cv * cos) * up;
    if (cx < 0 || cy < 0 || cx >= frame.width || cy >= frame.height) continue;

    const size = 4 * ((bu.p + bv.p) / 2) * up;
    seeds.push({ cx, cy, size, theta, score: scoreWindow(frame, cx, cy, size, theta) });
  }

  seeds.sort((a, b) => b.score - a.score);
  return seeds.slice(0, MAX_SEEDS);
}
