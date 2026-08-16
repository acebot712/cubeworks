import test from 'node:test';
import assert from 'node:assert/strict';
import { clusterStickers } from '../src/scan/cluster.js';

// The clusterer no longer names colours — it groups them. These tests assert the
// two properties the rest of the pipeline relies on: samples of one colour land
// in one group (purity), and each group holds exactly 16 (the error-correcting
// constraint). Naming is resolve.js's job and is covered in palette.test.js.
function purityOf(assign, truth) {
  const seen = {};
  for (let i = 0; i < assign.length; i++) (seen[truth[i]] ||= new Set()).add(assign[i]);
  return Object.values(seen).every((s) => s.size === 1);
}

const REF = {
  W: [242, 243, 245], Y: [255, 208, 40], G: [35, 177, 90],
  B: [44, 107, 232], R: [232, 64, 42], O: [255, 122, 26],
};
const KEYS = Object.keys(REF);

// deterministic pseudo-noise
function noise(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return (s / 2147483648 - 0.5) * 2;
  };
}

function makeSamples(cast, noiseAmp, seed) {
  const rnd = noise(seed);
  const rgbs = [];
  const truth = [];
  for (const k of KEYS) {
    for (let i = 0; i < 16; i++) {
      const [r, g, b] = REF[k];
      rgbs.push([
        Math.max(0, Math.min(255, r * cast[0] + rnd() * noiseAmp)),
        Math.max(0, Math.min(255, g * cast[1] + rnd() * noiseAmp)),
        Math.max(0, Math.min(255, b * cast[2] + rnd() * noiseAmp)),
      ]);
      truth.push(k);
    }
  }
  return { rgbs, truth };
}

test('clustering recovers all 96 stickers under neutral light', () => {
  const { rgbs, truth } = makeSamples([1, 1, 1], 10, 7);
  const { assign } = clusterStickers(rgbs);
  assert.ok(purityOf(assign, truth), 'samples of one colour split across groups');
});

test('clustering recovers all 96 stickers under a strong warm cast', () => {
  // warm indoor lighting: red boosted, blue crushed — the cast that makes
  // fixed HSV thresholds read white as yellow/orange
  const { rgbs, truth } = makeSamples([1.25, 1.0, 0.65], 10, 42);
  const { assign } = clusterStickers(rgbs);
  assert.ok(purityOf(assign, truth), 'samples of one colour split across groups');
});

test('clustering recovers all 96 stickers under a cool cast', () => {
  const { rgbs, truth } = makeSamples([0.8, 0.95, 1.3], 10, 99);
  const { assign } = clusterStickers(rgbs);
  assert.ok(purityOf(assign, truth), 'samples of one colour split across groups');
});

test('clustering always returns exactly 16 of each color', () => {
  const { rgbs } = makeSamples([1.15, 1.0, 0.8], 30, 3);
  const { assign, conf, centroids } = clusterStickers(rgbs);
  const counts = {};
  for (const k of assign) counts[k] = (counts[k] || 0) + 1;
  for (let k = 0; k < 6; k++) assert.equal(counts[k], 16, `group ${k}`);
  for (const c of conf) assert.ok(c >= 0 && c <= 1);
  assert.equal(centroids.length, 6);
});
