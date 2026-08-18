// Scoring one candidate window: how much does it look like a cube face?
//
// The structural claim: a cube face is a GRID, not stripes. It has seams in both
// axes at once, those seams are narrow (brightness recovers within a couple of
// px), its colours come from a small palette, and it ends: stepping away from
// the background on all four sides. Curtain folds and bare skin each satisfy at
// most one of those, which is what an earlier, looser scoring failed to require.
//
// Each component below answers one of those questions and is combined at the
// bottom. Every consumer only wants the final number, so that is all we return.
import { rgb2hsv } from '../classify.js';
import { clamp01, luma, chromaDiff, quadSampler, quadFromPose, cellCentre, gridLine } from './frame.js';

// --- cell means and interior uniformity ------------------------------------
// 16 cells, 5 taps each: centre plus four diagonal neighbours.
function readCells(px, size) {
  const means = new Array(16);
  let uniformity = 0;
  const d = size / 16; // diagonal tap offset
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const u0 = cellCentre(c, size), v0 = cellCentre(r, size);
      const taps = [
        px(u0, v0), px(u0 - d, v0 - d), px(u0 + d, v0 - d), px(u0 - d, v0 + d), px(u0 + d, v0 + d),
      ];
      let mr = 0, mg = 0, mb = 0;
      for (const t of taps) { mr += t[0]; mg += t[1]; mb += t[2]; }
      mr /= 5; mg /= 5; mb /= 5;
      means[r * 4 + c] = [mr, mg, mb];
      let mad = 0;
      for (const t of taps) mad += (Math.abs(t[0] - mr) + Math.abs(t[1] - mg) + Math.abs(t[2] - mb)) / 3;
      mad /= 5;
      uniformity += clamp01(1 - mad / 28);
    }
  }
  return { means, uniformity: uniformity / 16 };
}

// --- grid seam alignment ----------------------------------------------------
// Two requirements, each closing a false-positive route:
//  (a) the axes are accumulated separately and combined geometrically, so
//      single-axis periodicity (curtain folds, blinds, shelves, siding)
//      cannot fake a grid;
//  (b) each seam is probed at a NEAR flank as well as a FAR one. A real seam
//      is 1-2px wide, so the sticker colour has already returned at the near
//      flank (ratio ~1); a soft fold is still on its ramp there (~0.3).
//
// A seam is a DISCONTINUITY, not specifically a dark one. Measuring only
// "darker than both flanks" made the detector structurally blind to
// light-bodied stickerless cubes, whose seams are brighter than the stickers: // no threshold could have found them. Both directions now count.
function seamStrength(line, f1, f2) {
  const darker = Math.min(f1, f2) - line;  // dark seam between brighter stickers
  const lighter = line - Math.max(f1, f2); // light body between darker stickers
  return clamp01(Math.max(darker, lighter) / 13);
}

function seamAlignment(px, size) {
  const farOff = 0.10 * size;
  const nearOff = Math.max(1.4, 0.04 * size); // absolute floor: sensor blur is ~1px
  const gapFar = [0, 0], gapNear = [0, 0], falseGap = [0, 0]; // [0]=vertical, [1]=horizontal

  for (let k = 0; k < 4; k++) {
    const m = cellCentre(k, size); // position along the line being walked
    for (let ax = 0; ax < 2; ax++) {
      const horiz = ax === 1;
      const L = (a, b) => luma(horiz ? px(b, a) : px(a, b));
      for (let line = 1; line <= 3; line++) {
        const g = gridLine(line, size);
        const c0 = L(g, m);
        gapFar[ax] += seamStrength(c0, L(g - farOff, m), L(g + farOff, m));
        gapNear[ax] += seamStrength(c0, L(g - nearOff, m), L(g + nearOff, m));
      }
      // the same probe at CELL CENTRES, where a real face has no seam: this is
      // the baseline the real seams must beat
      for (let c = 0; c < 4; c++) {
        const cc = cellCentre(c, size);
        const c0 = L(cc, m);
        falseGap[ax] += seamStrength(c0, L(cc - farOff, m), L(cc + farOff, m));
      }
    }
  }

  const perAxis = [0, 0];
  for (let ax = 0; ax < 2; ax++) {
    gapFar[ax] /= 12; gapNear[ax] /= 12; falseGap[ax] /= 16;
    const rho = gapFar[ax] > 0.02 ? gapNear[ax] / gapFar[ax] : 0;
    const narrow = 0.20 + 0.80 * clamp01((rho - 0.55) / 0.30);
    perAxis[ax] = Math.max(0, gapFar[ax] - 1.3 * falseGap[ax]) * narrow;
  }
  // geometric mean, not min(): one strong axis and one dead axis scores ~0,
  // without the kink that min() would put in the hill-climber's path
  return Math.sqrt(perAxis[0] * perAxis[1]);
}

// How many of the 24 adjacent cell pairs differ in colour.
function colorPairCount(means) {
  let pairs = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      if (c < 3 && chromaDiff(means[r * 4 + c], means[r * 4 + c + 1]) > 46) pairs++;
      if (r < 3 && chromaDiff(means[r * 4 + c], means[(r + 1) * 4 + c]) > 46) pairs++;
    }
  }
  return pairs;
}

// --- four-edge silhouette ---------------------------------------------------
// A cube face ends. A patch of curtain does not: slide the window sideways and
// you find more curtain. Take the 3rd-best of the 4 edges so one occluded side
// (a hand, an adjacent face) does not veto an otherwise good detection.
function silhouette(px, inFrame, size) {
  const IN = 0.46 * size, OUT = 0.56 * size;
  const OFFSETS = [
    (t) => [t, -IN, t, -OUT], (t) => [t, IN, t, OUT],
    (t) => [-IN, t, -OUT, t], (t) => [IN, t, OUT, t],
  ];
  const edges = [];
  for (const place of OFFSETS) {
    let acc = 0, n = 0;
    for (let j = 0; j < 5; j++) {
      const t = ((j + 0.5) / 5 - 0.5) * 0.84 * size; // stay clear of the corners
      const [ui, vi, uo, vo] = place(t);
      // an off-frame outside sample would clamp to the border pixel and invent
      // a step, so drop it rather than counting it
      if (!inFrame(uo, vo)) continue;
      const ci = px(ui, vi), co = px(uo, vo);
      acc += clamp01(Math.max(chromaDiff(ci, co) / 48, Math.abs(luma(ci) - luma(co)) / 38));
      n++;
    }
    if (n >= 3) edges.push(acc / n);
  }
  edges.sort((a, b) => a - b);
  return edges.length ? edges[Math.max(0, edges.length - 3)] : 0;
}

// --- palette coherence ------------------------------------------------------
// Stickers come from a palette: the 16 cell means form a few tight, separated
// clusters. Skin and fabric spread across a continuum instead. Deliberately
// vacuous for a stickerless face (K=1 -> 1); it only punishes the continuum.
const TAU = 34; // under the 46 used for "different colour", over same-colour spread
const paletteDist = (a, b) => chromaDiff(a, b) + 0.30 * Math.abs(luma(a) - luma(b));

// single-link clustering of the 16 means: assign to the nearest centroid within
// TAU or start a new one, then merge any centroids that ended up too close
function clusterMeans(means) {
  const cent = [], count = [], assign = new Array(16);
  for (let i = 0; i < 16; i++) {
    let bi = -1, bd = Infinity;
    for (let j = 0; j < cent.length; j++) {
      const q = paletteDist(means[i], cent[j]);
      if (q < bd) { bd = q; bi = j; }
    }
    if (bi >= 0 && bd < TAU) {
      const n = count[bi];
      cent[bi] = [0, 1, 2].map((z) => (cent[bi][z] * n + means[i][z]) / (n + 1));
      count[bi] = n + 1; assign[i] = bi;
    } else {
      cent.push([...means[i]]); count.push(1); assign[i] = cent.length - 1;
    }
  }
  for (let a = 0; a < cent.length; a++) {
    for (let b = a + 1; b < cent.length; b++) {
      if (count[a] === 0 || count[b] === 0) continue;
      if (paletteDist(cent[a], cent[b]) >= TAU) continue;
      const na = count[a], nb = count[b];
      cent[a] = [0, 1, 2].map((z) => (cent[a][z] * na + cent[b][z] * nb) / (na + nb));
      count[a] = na + nb; count[b] = 0;
      for (let i = 0; i < 16; i++) if (assign[i] === b) assign[i] = a;
    }
  }
  return { cent, count, assign };
}

function paletteCoherence(means) {
  const { cent, count, assign } = clusterMeans(means);
  const live = [];
  for (let j = 0; j < cent.length; j++) if (count[j] > 0) live.push(j);
  const K = live.length;

  let spread = 0;
  for (let i = 0; i < 16; i++) spread += paletteDist(means[i], cent[assign[i]]);
  spread /= 16;

  let dmin = Infinity;
  for (let a = 0; a < live.length; a++) {
    for (let b = a + 1; b < live.length; b++) dmin = Math.min(dmin, paletteDist(cent[live[a]], cent[live[b]]));
  }

  return Math.min(
    K <= 6 ? 1 : Math.max(0, 1 - (K - 6) * 0.34), // a face shows at most 6 colours
    clamp01(1 - spread / 22),                     // clusters must be tight
    K >= 2 ? clamp01((dmin - 30) / 30) : 1,       // and mutually separated
  );
}

// --- sticker plausibility (PALETTE-FREE) ------------------------------------
// Deliberately no hue table. Cube brands ship pastel, neon, purple-for-blue and
// custom sticker sets; a fixed hue list silently rejects all of them. What
// actually separates a sticker from skin, wood, beige fabric and cardboard is
// not WHICH hue it is but that it is either vividly saturated or genuinely
// neutral-and-bright: never the muddy middle those materials occupy. The white
// test's saturation allowance only opens at high v, so a warm-cast white sticker
// (v 1.0, s 0.21) passes while beige curtain (v 0.84, s 0.20) does not.
function stickerGamut(means) {
  let plausible = 0;
  for (const m of means) {
    const [, s, v] = rgb2hsv(m[0], m[1], m[2]);
    if (v <= 0.15) continue;
    const whiteS = 0.15 + 0.30 * Math.max(0, v - 0.72);
    if (s < whiteS) { if (v > 0.62) plausible++; continue; }
    if (s < 0.40) continue; // muddy band -> not a sticker, whatever its hue
    plausible++;
  }
  return plausible / 16;
}

// Score an arbitrary quad [TL, TR, BR, BL]. Everything below is unchanged from
// the rotated-square version, the scorer only ever touches the image through
// px(u, v), so swapping in a homography generalises it to perspective for free.
// -> 0..1
export function scoreQuad(frame, q) {
  const sampler = quadSampler(frame, q);
  if (!sampler) return 0;
  const { px, inFrame, size } = sampler;
  const { means, uniformity } = readCells(px, size);

  const align = seamAlignment(px, size);
  // Seam alignment carries the scale/position evidence and must stay roughly
  // linear (saturating it lets half-scale windows tie the true one); colour
  // diversity is scale-blind so it only tops up (cap 0.30), and a stickerless
  // solved face contributes 0 there by construction.
  const grid = Math.min(1, 1.05 * align + Math.min(0.30, (colorPairCount(means) / 24) * 0.45));

  const edges = silhouette(px, inFrame, size);
  const gamut = stickerGamut(means);
  const palette = paletteCoherence(means);

  // The gate reads `align`, not `grid`: colourful clutter with no grid must not
  // open it, and a blank wall is perfectly uniform but has no seams at all.
  const gate = 0.20 + 0.80 * Math.min(1, align / 0.30);
  return (0.30 * uniformity + 0.40 * grid + 0.30 * edges) * (0.35 + 0.65 * gamut) * (0.55 + 0.45 * palette) * gate;
}

// -> 0..1, for the rotated-square parameterisation the coarse stage still uses
export function scoreWindow(frame, cx, cy, size, theta) {
  return scoreQuad(frame, quadFromPose(cx, cy, size, theta));
}
