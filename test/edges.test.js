import test from 'node:test';
import assert from 'node:assert';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { randomScramble } from '../src/cube/scramble.js';
import { mulberry32 } from '../src/cube/rng.js';
import { solveCenters } from '../src/solver/centers.js';
import { solveEdges } from '../src/solver/edges.js';
import { wingsFromState, pairedCount } from '../src/solver/tables.js';

test('edge pairing after centers on 30 random scrambles', () => {
  let totalMoves = 0;
  let maxMs = 0;
  for (let i = 0; i < 30; i++) {
    const rand = mulberry32(2000 + i);
    const scr = randomScramble(45, rand);
    const scrambled = applyMoves(solvedState(), scr);
    const centers = solveCenters(scrambled);
    const t0 = Date.now();
    const { moves, state } = solveEdges(centers.state);
    const dt = Date.now() - t0;
    maxMs = Math.max(maxMs, dt);
    assert.equal(pairedCount(wingsFromState(state)), 12, 'not fully paired, seed ' + i);
    const check = applyMoves(centers.state, moves);
    assert.equal(pairedCount(wingsFromState(check)), 12, 'reported moves fail, seed ' + i);
    totalMoves += moves.length;
  }
  console.log('avg edge moves:', (totalMoves / 30).toFixed(1), 'max ms:', maxMs);
});
