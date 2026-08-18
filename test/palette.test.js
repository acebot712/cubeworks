// The scanner must not depend on which cube you own.
//
// Brands ship very different palettes: pastel sets, "bright" stickerless,
// purple in place of blue, pink in place of red, custom stickers. None of the
// six colours below are the canonical ones, and the pipeline must still resolve
// the cube exactly, because it clusters by structure (16 stickers per colour,
// piece geometry names the faces), never by matching hues to a table.
//
// On what is knowable: from a SCRAMBLED cube the colour-to-face labelling is
// determined only up to the 24 whole-cube rotations. That information is not
// present in the stickers, and it is not needed, the solver works in position
// space, so any of the 24 gives a correct solution. What must hold is that one
// physical colour lands in one class (purity) and that the result is a real,
// non-mirrored cube (validity + chirality).
import test from 'node:test';
import assert from 'node:assert/strict';
import { clusterStickers } from '../src/scan/cluster.js';
import { labelClusters } from '../src/scan/resolve.js';
import { FACES } from '../src/cube/geometry.js';
import { parseMoves } from '../src/cube/notation.js';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { validateState } from '../src/cube/validate.js';
import { describeProblems } from '../src/ui/problemText.js';
import { solve4x4 } from '../src/solver/index.js';
import { COLOR_OF_FACE, FACE_OF_COLOR } from '../src/state/colors.js';

const SCRAMBLE = parseMoves("R U2 Fw' D L2 Bw R' Uw D2 F L' B2 Rw U' F2 Dw R2 B");

// six wildly non-standard palettes, keyed by FACE (not by colour name)
const PALETTES = {
  standard: { U: [242, 243, 245], R: [232, 64, 42], F: [35, 177, 90], D: [255, 208, 40], L: [255, 122, 26], B: [44, 107, 232] },
  // pastel/macaron set: everything washed out
  pastel: { U: [246, 244, 238], R: [239, 138, 138], F: [150, 214, 160], D: [250, 232, 150], L: [246, 186, 130], B: [140, 176, 232] },
  // "purple for blue, pink for red" custom sticker set
  custom: { U: [240, 240, 244], R: [226, 74, 140], F: [64, 190, 120], D: [246, 220, 70], L: [246, 140, 44], B: [126, 74, 214] },
  // neon / UV set, very high saturation
  neon: { U: [250, 250, 250], R: [255, 30, 60], F: [40, 255, 90], D: [250, 255, 20], L: [255, 140, 0], B: [30, 90, 255] },
};

function lcg(seed) {
  let s = seed;
  return () => { s = (s * 1103515245 + 12345) % 2147483648; return (s / 2147483648 - 0.5) * 2; };
}

// build the 96 RGB samples a perfect scan of `state` would produce on `palette`,
// with sensor noise and an optional whole-scene colour cast
function samples(state, palette, { noise = 8, cast = [1, 1, 1], seed = 5 } = {}) {
  const rnd = lcg(seed);
  const out = new Array(96);
  for (let i = 0; i < 96; i++) {
    const face = FACES[state[i]];
    const [r, g, b] = palette[face];
    out[i] = [
      Math.max(0, Math.min(255, r * cast[0] + rnd() * noise)),
      Math.max(0, Math.min(255, g * cast[1] + rnd() * noise)),
      Math.max(0, Math.min(255, b * cast[2] + rnd() * noise)),
    ];
  }
  return out;
}

// resolve samples back to an engine state the way App.jsx does
function resolve(rgbs) {
  const cl = clusterStickers(rgbs);
  const { faceOfCluster } = labelClusters(cl.assign, cl.centroids);
  const st = new Uint8Array(96);
  for (let i = 0; i < 96; i++) st[i] = FACES.indexOf(faceOfCluster[cl.assign[i]]);
  return { state: st, cl };
}

// every sticker of one physical colour must land in exactly one class
function purity(cl, truth) {
  const seen = {};
  for (let i = 0; i < 96; i++) (seen[truth[i]] ||= new Set()).add(cl.assign[i]);
  return Object.values(seen).every((s) => s.size === 1);
}

// same cube as truth, up to relabelling colours by a whole-cube rotation
function sameCubeUpToRotation(state, truth) {
  const map = new Map(), rev = new Map();
  for (let i = 0; i < 96; i++) {
    const a = truth[i], b = state[i];
    if (map.has(a) && map.get(a) !== b) return false;
    if (rev.has(b) && rev.get(b) !== a) return false;
    map.set(a, b); rev.set(b, a);
  }
  return map.size === 6;
}

for (const [name, palette] of Object.entries(PALETTES)) {
  test(`resolves a scrambled cube on the "${name}" palette`, () => {
    const truth = applyMoves(solvedState(), SCRAMBLE);
    const { state, cl } = resolve(samples(truth, palette));
    assert.ok(purity(cl, truth), `${name}: stickers of one colour split across classes`);
    const v = validateState(state);
    assert.ok(v.ok, `${name}: ${describeProblems(v.problems).join(' / ')}`);
    assert.ok(sameCubeUpToRotation(state, truth), `${name}: not the same cube`);
  });

  test(`resolves the "${name}" palette under a strong warm cast`, () => {
    const truth = applyMoves(solvedState(), SCRAMBLE);
    const { state, cl } = resolve(samples(truth, palette, { cast: [1.22, 1.0, 0.7], seed: 21 }));
    assert.ok(purity(cl, truth), `${name} + cast: classes impure`);
    assert.ok(validateState(state).ok, `${name} + cast: invalid`);
    assert.ok(sameCubeUpToRotation(state, truth), `${name} + cast: not the same cube`);
  });

  test(`the "${name}" palette resolves to a cube the solver actually solves`, () => {
    const truth = applyMoves(solvedState(), SCRAMBLE);
    const { state } = resolve(samples(truth, palette, { seed: 77 }));
    const sol = solve4x4(state);
    const end = applyMoves(state, sol.phases.flatMap((p) => p.moves));
    for (let f = 0; f < 6; f++) {
      const first = end[f * 16];
      for (let i = 1; i < 16; i++) assert.equal(end[f * 16 + i], first, `${name}: face ${FACES[f]} not uniform`);
    }
  });
}

test('a solved cube resolves on a non-standard palette', () => {
  const truth = solvedState();
  const { state, cl } = resolve(samples(truth, PALETTES.custom, { seed: 9 }));
  assert.ok(purity(cl, truth));
  assert.ok(sameCubeUpToRotation(state, truth));
});

test('labelling is driven by piece geometry, not by hue similarity', () => {
  // Up is PURPLE and Back is WHITE, the inverse of the canonical scheme.
  const inverted = {
    U: [126, 74, 214], R: [232, 64, 42], F: [35, 177, 90],
    D: [255, 208, 40], L: [255, 122, 26], B: [240, 240, 244],
  };
  const truth = applyMoves(solvedState(), SCRAMBLE);
  const { state, cl } = resolve(samples(truth, inverted, { seed: 33 }));
  assert.ok(purity(cl, truth));
  assert.ok(validateState(state).ok);
  assert.ok(sameCubeUpToRotation(state, truth));
});

test('centroids report the cube\'s own colours for display', () => {
  const truth = applyMoves(solvedState(), SCRAMBLE);
  const cl = clusterStickers(samples(truth, PALETTES.custom, { noise: 4, seed: 12 }));
  const { faceOfCluster } = labelClusters(cl.assign, cl.centroids);
  for (let k = 0; k < 6; k++) {
    const facesInCluster = new Set();
    for (let i = 0; i < 96; i++) if (cl.assign[i] === k) facesInCluster.add(FACES[truth[i]]);
    assert.equal(facesInCluster.size, 1, `cluster ${k} mixes faces`);
    const want = PALETTES.custom[[...facesInCluster][0]];
    for (let d = 0; d < 3; d++) {
      assert.ok(Math.abs(cl.centroids[k][d] - want[d]) < 12, `centroid ${k} ch ${d}: ${cl.centroids[k][d]} vs ${want[d]}`);
    }
  }
  assert.equal(new Set(faceOfCluster).size, 6, 'labelling is not a bijection');
});

test('every colour letter still appears exactly 16 times', () => {
  const truth = applyMoves(solvedState(), SCRAMBLE);
  const cl = clusterStickers(samples(truth, PALETTES.pastel, { seed: 4 }));
  const { faceOfCluster } = labelClusters(cl.assign, cl.centroids);
  const counts = {};
  for (let i = 0; i < 96; i++) {
    const letter = COLOR_OF_FACE[faceOfCluster[cl.assign[i]]];
    counts[letter] = (counts[letter] || 0) + 1;
  }
  for (const letter of Object.keys(FACE_OF_COLOR)) assert.equal(counts[letter], 16, `letter ${letter}`);
});
