import test from 'node:test';
import assert from 'node:assert';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { randomScramble } from '../src/cube/scramble.js';
import { mulberry32 } from '../src/cube/rng.js';
import { solveCenters } from '../src/solver/centers.js';
import { centersFromState } from '../src/solver/tables.js';

function centersSolved(state) {
  const c = centersFromState(state);
  for (let f = 0; f < 6; f++) for (let k = 0; k < 4; k++) if (c[f * 4 + k] !== f) return false;
  return true;
}

test('centers solver on 30 random scrambles', () => {
  let totalMoves = 0;
  for (let i = 0; i < 30; i++) {
    const rand = mulberry32(1000 + i);
    const scr = randomScramble(45, rand);
    const scrambled = applyMoves(solvedState(), scr);
    const t0 = Date.now();
    const { moves, state } = solveCenters(scrambled);
    const dt = Date.now() - t0;
    assert.ok(centersSolved(state), 'centers not solved for seed ' + i);
    // verify moves independently
    const check = applyMoves(scrambled, moves);
    assert.ok(centersSolved(check), 'reported moves do not solve centers, seed ' + i);
    totalMoves += moves.length;
    if (dt > 8000) console.log('slow seed', i, dt + 'ms');
  }
  console.log('avg center moves:', (totalMoves / 30).toFixed(1));
});
