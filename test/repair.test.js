import test from 'node:test';
import assert from 'node:assert/strict';
import { rotGrid, repairState } from '../src/cube/repair.js';
import { FACES } from '../src/cube/geometry.js';
import { parseMoves } from '../src/cube/notation.js';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { CAPTURE_STEPS, CAPTURE_MAPS, SWAPPED_RL_MAPS } from '../src/scan/orientations.js';
import { COLOR_OF_FACE } from '../src/state/colors.js';

const SCRAMBLE = parseMoves("R U2 Fw' D L2 Bw R' Uw D2 F L' B2 Rw U' F2 Dw' L Uw2 B R2 F' Lw D' U R Fw2 L2 B' Dw R'");

function capturesFor(state, { maps = CAPTURE_MAPS, conf = 0.9 } = {}) {
  const out = {};
  for (const step of CAPTURE_STEPS) {
    const map = maps[step.key];
    const grid = new Array(16);
    const cf = new Array(16);
    for (let i = 0; i < 16; i++) {
      grid[i] = COLOR_OF_FACE[FACES[state[map[i]]]];
      cf[i] = conf;
    }
    out[step.key] = { grid, conf: cf };
  }
  return out;
}

function colorsOf(state) {
  return [...state].map((c) => COLOR_OF_FACE[FACES[c]]);
}

const SCRAMBLED = applyMoves(solvedState(), SCRAMBLE);
const TRUTH = colorsOf(SCRAMBLED);

test('rotGrid: four quarter turns is the identity', () => {
  const g = [...Array(16).keys()].map(String);
  assert.deepEqual(rotGrid(g, 4), g);
  assert.deepEqual(rotGrid(g, 0), g);
  // one clockwise turn puts the bottom-left cell (12) at top-left
  assert.equal(rotGrid(g, 1)[0], '12');
  assert.equal(rotGrid(g, 1)[3], '0');
  assert.deepEqual(rotGrid(rotGrid(g, 1), 3), g);
});

test('valid captures repair to themselves at zero cost', () => {
  const r = repairState(capturesFor(SCRAMBLED), CAPTURE_MAPS, SWAPPED_RL_MAPS);
  assert.ok(r);
  assert.equal(r.rlSwapped, false);
  assert.deepEqual(r.fixes, []);
  for (const k of Object.keys(r.rotations)) assert.equal(r.rotations[k], 0);
  assert.deepEqual(r.colors, TRUTH);
});

test('recovers a face captured at the wrong rotation', () => {
  const caps = capturesFor(SCRAMBLED);
  caps.F = { grid: rotGrid(caps.F.grid, 1), conf: caps.F.conf };
  const r = repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS);
  assert.ok(r, 'expected a repair');
  assert.deepEqual(r.colors, TRUTH);
  assert.equal(r.rotations.F, 3, 'F should be rotated back');
});

test('recovers an R/L yaw swap', () => {
  const caps = capturesFor(SCRAMBLED, { maps: { ...CAPTURE_MAPS, ...SWAPPED_RL_MAPS } });
  const r = repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS);
  assert.ok(r, 'expected a repair');
  assert.equal(r.rlSwapped, true);
  assert.deepEqual(r.colors, TRUTH);
});

test('recovers a rotation and an R/L swap together', () => {
  const caps = capturesFor(SCRAMBLED, { maps: { ...CAPTURE_MAPS, ...SWAPPED_RL_MAPS } });
  caps.U = { grid: rotGrid(caps.U.grid, 2), conf: caps.U.conf };
  const r = repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS);
  assert.ok(r, 'expected a repair');
  assert.equal(r.rlSwapped, true);
  assert.equal(r.rotations.U, 2);
  assert.deepEqual(r.colors, TRUTH);
});

test('corrects a single misread sticker flagged low-confidence', () => {
  const caps = capturesFor(SCRAMBLED);
  // find a red sticker on the F capture and misread it as orange
  const i = caps.F.grid.findIndex((c) => c === 'R');
  assert.ok(i >= 0, 'scramble should expose a red sticker on F');
  caps.F.grid[i] = 'O';
  caps.F.conf[i] = 0.4;
  const r = repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS);
  assert.ok(r, 'expected a repair');
  assert.equal(r.fixes.length, 1);
  assert.equal(r.fixes[0].from, 'O');
  assert.equal(r.fixes[0].to, 'R');
  assert.deepEqual(r.colors, TRUTH);
});

test('returns null for unrepairable garbage, quickly', () => {
  const caps = capturesFor(SCRAMBLED);
  // scramble the colors themselves into nonsense
  const letters = ['W', 'Y', 'G', 'B', 'R', 'O'];
  let n = 7;
  for (const k of Object.keys(caps)) {
    caps[k] = {
      grid: caps[k].grid.map(() => { n = (n * 1103515245 + 12345) % 2147483648; return letters[n % 6]; }),
      conf: new Array(16).fill(0.5),
    };
  }
  const t0 = performance.now();
  const r = repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS);
  const dt = performance.now() - t0;
  assert.equal(r, null);
  assert.ok(dt < 3000, `took ${dt.toFixed(0)}ms`);
});

test('returns null when faces are missing', () => {
  const caps = capturesFor(SCRAMBLED);
  delete caps.B;
  assert.equal(repairState(caps, CAPTURE_MAPS, SWAPPED_RL_MAPS), null);
});
