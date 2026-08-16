import test from 'node:test';
import assert from 'node:assert';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { randomScramble } from '../src/cube/scramble.js';
import { mulberry32 } from '../src/cube/rng.js';
import { solve4x4, isSolved } from '../src/solver/index.js';
import { validateState } from '../src/cube/validate.js';

test('full pipeline solves 25 random scrambles (verified)', () => {
  let total = 0;
  let maxMoves = 0;
  let maxMs = 0;
  for (let i = 0; i < 25; i++) {
    const rand = mulberry32(9000 + i);
    const scrambled = applyMoves(solvedState(), randomScramble(45, rand));
    assert.ok(validateState(scrambled).ok, 'scrambled state valid, seed ' + i);
    const t0 = Date.now();
    const { phases, totalMoves } = solve4x4(scrambled);
    const dt = Date.now() - t0;
    const flat = phases.flatMap((p) => p.moves);
    const end = applyMoves(scrambled, flat);
    assert.ok(isSolved(end), 'independently replayed to solved, seed ' + i);
    total += totalMoves;
    maxMoves = Math.max(maxMoves, totalMoves);
    maxMs = Math.max(maxMs, dt);
  }
  console.log('avg moves:', (total / 25).toFixed(1), 'max moves:', maxMoves, 'max ms:', maxMs);
});

test('solving an already-solved cube yields no moves', () => {
  const { totalMoves } = solve4x4(solvedState());
  assert.equal(totalMoves, 0);
});
