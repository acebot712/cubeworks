import test from 'node:test';
import assert from 'node:assert/strict';
import { detectFace, detectFaceInRoi } from '../src/scan/detect/search.js';
import { scoreWindow } from '../src/scan/detect/score.js';
import { createTracker, ACQUIRE_SCORE as ACQ } from '../src/scan/detect/tracker.js';

const REF = {
  W: [242, 243, 245], Y: [255, 208, 40], G: [35, 177, 90],
  B: [44, 107, 232], R: [232, 64, 42], O: [255, 122, 26],
};

function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
const KEYS = Object.keys(REF);

function lcg(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return (s / 2147483648 - 0.5) * 2;
  };
}

function makeFrame(w, h, bg) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const rgb = typeof bg === 'function' ? bg(i % w, (i / w) | 0) : bg;
    data[i * 4] = rgb[0]; data[i * 4 + 1] = rgb[1]; data[i * 4 + 2] = rgb[2]; data[i * 4 + 3] = 255;
  }
  return { data, width: w, height: h };
}

// draw a 4x4 face (rotated square) into the frame, pixel by pixel
function drawFace(frame, cx, cy, size, theta, colors16, opts = {}) {
  const gapFrac = opts.gapFrac ?? 0.06;
  const gapLum = opts.gapLum ?? 25;
  const noise = opts.noise ?? 6;
  const cast = opts.cast ?? [1, 1, 1];
  const rnd = lcg(opts.seed ?? 1);
  const cos = Math.cos(-theta), sin = Math.sin(-theta);
  const half = size / 2;
  const bound = Math.ceil(half * 1.5);
  for (let y = Math.max(0, cy - bound | 0); y < Math.min(frame.height, cy + bound); y++) {
    for (let x = Math.max(0, cx - bound | 0); x < Math.min(frame.width, cx + bound); x++) {
      // inverse-rotate into square-local coords
      const dx = x - cx, dy = y - cy;
      const u = dx * cos - dy * sin, v = dx * sin + dy * cos;
      if (u < -half || u >= half || v < -half || v >= half) continue;
      const fu = (u + half) / size * 4, fv = (v + half) / size * 4;
      const cu = fu - Math.floor(fu), cvv = fv - Math.floor(fv);
      const inGap = cu < gapFrac || cu > 1 - gapFrac || cvv < gapFrac || cvv > 1 - gapFrac;
      let rgb;
      if (inGap) rgb = [gapLum, gapLum, gapLum];
      else {
        const col = REF[colors16[Math.min(3, Math.floor(fv)) * 4 + Math.min(3, Math.floor(fu))]];
        rgb = [col[0] * cast[0], col[1] * cast[1], col[2] * cast[2]];
      }
      const i = (y * frame.width + x) * 4;
      frame.data[i] = Math.max(0, Math.min(255, rgb[0] + rnd() * noise));
      frame.data[i + 1] = Math.max(0, Math.min(255, rgb[1] + rnd() * noise));
      frame.data[i + 2] = Math.max(0, Math.min(255, rgb[2] + rnd() * noise));
    }
  }
}

function scrambledColors(seed) {
  const rnd = lcg(seed);
  return [...Array(16)].map(() => KEYS[Math.floor((rnd() * 0.5 + 0.5) * 6) % 6]);
}

const W = 192, H = 108;

test('detects a centered scrambled face', () => {
  const f = makeFrame(W, H, [107, 111, 118]);
  const size = 0.6 * H, cx = W / 2, cy = H / 2;
  drawFace(f, cx, cy, size, 0, scrambledColors(3), { seed: 5 });
  const d = detectFace(f);
  assert.ok(d.score >= ACQ, `score ${d.score}`);
  assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.05 * size, `center off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
  assert.ok(Math.abs(d.size - size) / size < 0.12, `size ${d.size} vs ${size}`);
  assert.ok(Math.abs(d.theta) < 4 * Math.PI / 180, `theta ${d.theta}`);
});

test('detects an off-center small face under a warm cast', () => {
  const f = makeFrame(W, H, [95, 99, 106]);
  const size = 0.32 * H, cx = W * 0.72, cy = H * 0.33;
  drawFace(f, cx, cy, size, 0, scrambledColors(9), { cast: [1.15, 1.0, 0.82], noise: 8, seed: 11 });
  const d = detectFace(f);
  assert.ok(d.score >= ACQ, `score ${d.score}`);
  assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.08 * size + 2, `center off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
  assert.ok(Math.abs(d.size - size) / size < 0.15, `size ${d.size} vs ${size}`);
});

test('recovers a 10-degree rotation', () => {
  const f = makeFrame(W, H, [107, 111, 118]);
  const size = 0.55 * H, cx = W / 2 + 8, cy = H / 2 - 4;
  const theta = 10 * Math.PI / 180;
  drawFace(f, cx, cy, size, theta, scrambledColors(7), { seed: 13 });
  const d = detectFace(f);
  assert.ok(d.score >= ACQ, `score ${d.score}`);
  assert.ok(Math.abs(d.theta - theta) < 4 * Math.PI / 180, `theta ${d.theta} vs ${theta}`);
});

test('detects a stickerless solved face (seams only, no color pairs)', () => {
  const f = makeFrame(W, H, [90, 94, 100]);
  const size = 0.6 * H, cx = W / 2, cy = H / 2;
  // uniform green with genuinely DARKER seams. (An earlier version of this
  // fixture used gapLum 120 against green whose luma is 115 — the "seam" was
  // brighter than the sticker, so the test passed only via a shrunken rotated
  // window. Hence the tight size/theta assertions below.)
  drawFace(f, cx, cy, size, 0, new Array(16).fill('G'), { gapLum: 88, gapFrac: 0.06, noise: 4, seed: 17 });
  const d = detectFace(f);
  assert.ok(d && d.score >= ACQ, `score ${d && d.score}`);
  assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.06 * size + 2, `center off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
  assert.ok(Math.abs(d.size - size) / size < 0.10, `size ${d.size} vs ${size}`);
  assert.ok(Math.abs(d.theta) < 4 * Math.PI / 180, `theta ${d.theta}`);
});

test('rejects cube-free frames (flat, gradient, white wall)', () => {
  const flat = makeFrame(W, H, [107, 111, 118]);
  const grad = makeFrame(W, H, (x, y) => [60 + x * 0.5, 65 + y * 0.4, 75 + x * 0.3]);
  const wall = makeFrame(W, H, [235, 233, 228]);
  for (const [name, f] of [['flat', flat], ['gradient', grad], ['white wall', wall]]) {
    const d = detectFace(f);
    assert.ok(!d || d.score < ACQ, `${name} scored ${d && d.score}`);
  }
});

test('tracker hysteresis: acquire, grace through occlusion, lose, re-acquire', () => {
  const cube = makeFrame(W, H, [107, 111, 118]);
  drawFace(cube, W / 2, H / 2, 0.6 * H, 0, scrambledColors(3), { seed: 5 });
  const blank = makeFrame(W, H, [107, 111, 118]);
  const moved = makeFrame(W, H, [107, 111, 118]);
  drawFace(moved, W * 0.65, H * 0.45, 0.5 * H, 0, scrambledColors(3), { seed: 5 });

  const tr = createTracker();
  assert.equal(tr.update(cube).found, false); // 1st good tick (anchors)
  assert.equal(tr.update(cube).found, false); // 2nd
  assert.equal(tr.update(cube).found, true);  // 3rd -> acquired
  // occlusion grace: 3 blank ticks keep the quad
  assert.equal(tr.update(blank).found, true);
  assert.equal(tr.update(blank).found, true);
  assert.equal(tr.update(blank).found, true);
  assert.equal(tr.update(blank).found, false); // 4th -> lost
  // re-acquire at the new position
  tr.update(moved);
  tr.update(moved);
  const r = tr.update(moved);
  assert.equal(r.found, true);
  assert.ok(Math.hypot(r.quad.cx - W * 0.65, r.quad.cy - H * 0.45) < 0.08 * 0.5 * H + 2);
});

// ---- the real-world false positive: beige curtain folds + bare skin ----
// Reported 2026-07-29: the detector locked onto curtains and a shoulder on the
// left of frame while ignoring the cube held on the right.

const bgNoise = lcg(99);

// periodic VERTICAL folds with soft edges and no horizontal structure at all
function curtainAt(x, y) {
  const P = 13;
  const ph = ((x % P) + P) % P;
  const dFold = Math.abs(ph - 6.5);
  const shade = 1 - 0.34 * (1 - smoothstep(0.6, 4.2, dFold));
  const vig = 1 - 0.12 * smoothstep(0, H, y);
  const n = bgNoise() * 3;
  return [214 * shade * vig + n, 199 * shade * vig + n, 172 * shade * vig + n];
}

// bare shoulder: smooth skin-tone gradient, locally uniform everywhere
function skinAt(x, y) {
  const g = 1 - 0.18 * smoothstep(0, 90, x) + 0.10 * smoothstep(50, 108, y);
  const n = bgNoise() * 2.5;
  return [205 * g + n, 150 * g + n, 120 * g + n];
}

function curtainSkinBg(x, y) {
  if (x < 92) return (y > 44 + 0.25 * x && x < 78) ? skinAt(x, y) : curtainAt(x, y);
  return [104, 108, 116];
}

test('rejects beige curtain folds (single-axis periodicity)', () => {
  const d = detectFace(makeFrame(W, H, curtainAt));
  assert.ok(!d || d.score < ACQ, `curtain scored ${d && d.score}`);
});

test('rejects a smooth skin-tone gradient', () => {
  const d = detectFace(makeFrame(W, H, skinAt));
  assert.ok(!d || d.score < ACQ, `skin scored ${d && d.score}`);
});

test('rejects curtain plus skin together', () => {
  const d = detectFace(makeFrame(W, H, curtainSkinBg));
  assert.ok(!d || d.score < ACQ, `curtain+skin scored ${d && d.score}`);
});

test('finds the cube on the right, not the curtain and skin on the left', () => {
  const f = makeFrame(W, H, curtainSkinBg);
  const cx = 140, cy = 54, size = 0.58 * H;
  drawFace(f, cx, cy, size, 0, scrambledColors(3), { seed: 5 });
  const d = detectFace(f);
  assert.ok(d && d.score >= ACQ, `score ${d && d.score}`);
  assert.ok(d.cx > W / 2, `locked onto the left half at cx ${d.cx}`);
  assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.08 * size, `center off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
  assert.ok(Math.abs(d.size - size) / size < 0.12, `size ${d.size} vs ${size}`);
});

test('tracker never acquires on curtain and skin over 10 ticks', () => {
  const f = makeFrame(W, H, curtainSkinBg);
  const tr = createTracker();
  for (let i = 0; i < 10; i++) {
    assert.equal(tr.update(f).found, false, `acquired at tick ${i}`);
  }
});

test('detects a stickerless solved WHITE face', () => {
  const f = makeFrame(W, H, [70, 74, 80]);
  const size = 0.6 * H, cx = W / 2, cy = H / 2;
  drawFace(f, cx, cy, size, 0, new Array(16).fill('W'), { gapLum: 150, gapFrac: 0.06, noise: 4, seed: 23 });
  const d = detectFace(f);
  assert.ok(d && d.score >= ACQ, `score ${d && d.score}`);
  assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.06 * size + 2, `center off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
});

// Seams are DISCONTINUITIES, not necessarily dark ones. A light-bodied
// stickerless cube (very common) has seams BRIGHTER than its stickers; the
// original darkness-only seam test made those cubes structurally undetectable.
test('detects a light-bodied cube (seams brighter than the stickers)', () => {
  for (const gapLum of [190, 215, 240]) {
    const f = makeFrame(W, H, [104, 108, 116]);
    const cx = W / 2, cy = H / 2, size = 0.55 * H;
    drawFace(f, cx, cy, size, 0.08, scrambledColors(3), { gapLum, gapFrac: 0.07, seed: 31 });
    const d = detectFace(f);
    assert.ok(d && d.score >= ACQ, `body luma ${gapLum} scored ${d && d.score}`);
    assert.ok(Math.hypot(d.cx - cx, d.cy - cy) < 0.08 * size, `body luma ${gapLum} centre off by ${Math.hypot(d.cx - cx, d.cy - cy)}`);
  }
});

test('detects a light-bodied SOLVED face (one colour, bright seams)', () => {
  const f = makeFrame(W, H, [70, 74, 80]);
  const cx = W / 2, cy = H / 2, size = 0.6 * H;
  drawFace(f, cx, cy, size, 0, new Array(16).fill('R'), { gapLum: 225, gapFrac: 0.07, noise: 4, seed: 37 });
  const d = detectFace(f);
  assert.ok(d && d.score >= ACQ, `score ${d && d.score}`);
});

// The coarse scan's cost IS "how many candidate windows does it score", so
// measure it in that unit rather than in milliseconds. A wall-clock budget
// fails for reasons that have nothing to do with the detector: this assertion
// used to read `dt < 30` and tripped at 57ms while the MLX trainer was running,
// having passed at 25ms on the same commit minutes earlier.
//
// scoreWindow is a fair unit because it takes a fixed number of taps whatever
// the window size — measured 25-31us across a 7x range of sizes.
//
// Both halves are timed alternately and each is reduced by MIN across rounds.
// Preemption only ever adds time, so the fastest round of each approximates
// what that half costs with the CPU to itself, and the ratio of the two minima
// is close to load-independent. Measured: 637-674 idle, 602-661 with twelve
// spinners and the MLX trainer competing. (The median across rounds is much
// worse here — it reached 907 under that load, which overlaps the regression
// band below and would make the budget unsettable.)
//
// The budget sits between the measured baseline and a real regression: widening
// the seed stage's angle sweep from 5 steps to 11 — a plausible change someone
// might make for robustness — costs 933-1043, and trips this.
const WINDOW_SCORE_BUDGET = 850;   // baseline ~655, ceiling seen 674

test('coarse scan stays within budget', () => {
  const f = makeFrame(W, H, [107, 111, 118]);
  const cx = W / 2, cy = H / 2, size = 0.6 * H;
  drawFace(f, cx, cy, size, 0, scrambledColors(3), { seed: 5 });

  // Warm the JIT first, or the early rounds time the optimiser rather than the
  // search — cold, detectFace costs several times its steady-state.
  for (let i = 0; i < 3; i++) detectFace(f);
  for (let i = 0; i < 3000; i++) scoreWindow(f, cx, cy, size, 0);

  const BATCH = 600, ROUNDS = 15;
  let full = Infinity, unit = Infinity;
  for (let r = 0; r < ROUNDS; r++) {
    let t = performance.now();
    detectFace(f);
    full = Math.min(full, performance.now() - t);

    t = performance.now();
    for (let i = 0; i < BATCH; i++) scoreWindow(f, cx, cy, size, 0);
    unit = Math.min(unit, (performance.now() - t) / BATCH);
  }
  const cost = full / unit;

  assert.ok(cost < WINDOW_SCORE_BUDGET,
    `coarse scan cost ${Math.round(cost)} window-scores, budget ${WINDOW_SCORE_BUDGET} `
    + `(detectFace ${full.toFixed(2)}ms, scoreWindow ${(unit * 1000).toFixed(2)}us)`);
});

// ---- ROI-restricted acquisition ----
// Measured on real captured frames: giving the search a rough box makes it
// 10-17x faster AND more accurate, because the comb no longer has to resolve a
// small face against a whole room. Any box source works — a model, or the
// user's own hand-placed frame.
test('a rough box makes the search faster and no worse', () => {
  const f = makeFrame(W, H, curtainSkinBg);
  const cx = 140, cy = 54, size = 0.34 * H;
  drawFace(f, cx, cy, size, 0.1, scrambledColors(3), { seed: 5 });

  const full = detectFace(f);
  // a deliberately sloppy box: off-centre and 25% too large, as a model's would be
  const box = [cx - size * 0.72, cy - size * 0.66, cx + size * 0.66, cy + size * 0.72];
  const roi = detectFaceInRoi(f, box);

  assert.ok(roi, 'ROI search found nothing');
  assert.ok(roi.score >= ACQ, `ROI score ${roi.score}`);
  assert.ok(Math.hypot(roi.cx - cx, roi.cy - cy) < 0.12 * size, `centre off by ${Math.hypot(roi.cx - cx, roi.cy - cy)}`);
  if (full) assert.ok(roi.score >= full.score - 0.05, `ROI ${roi.score} much worse than full ${full.score}`);
});

test('a box containing no cube yields nothing usable', () => {
  const f = makeFrame(W, H, curtainSkinBg);
  drawFace(f, 140, 54, 0.34 * H, 0, scrambledColors(3), { seed: 5 });
  // point the box at the curtain half instead
  const roi = detectFaceInRoi(f, [8, 12, 78, 96]);
  assert.ok(!roi || roi.score < ACQ, `curtain ROI scored ${roi && roi.score}`);
});
