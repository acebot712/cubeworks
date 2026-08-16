import test from 'node:test';
import assert from 'node:assert';
import { solvedState, applyMoves } from '../src/cube/state.js';
import { centersFromState, wingsFromState, pairedCount } from '../src/solver/tables.js';
import { extract3x3, analyze3x3 } from '../src/solver/three.js';
import { OLL_PARITY_ALG, PLL_PARITY_ALG } from '../src/solver/algs.js';

function centersSolved(state) {
  const c = centersFromState(state);
  for (let i = 0; i < 24; i++) if (c[i] !== ((i / 4) | 0)) return false;
  return true;
}

test('OLL parity alg preserves reduction and toggles edge flip parity', () => {
  const s = applyMoves(solvedState(), OLL_PARITY_ALG);
  assert.ok(centersSolved(s), 'centers');
  assert.equal(pairedCount(wingsFromState(s)), 12, 'pairing');
  const info = analyze3x3(extract3x3(s));
  assert.equal(info.edgeParityOdd, true, 'should be odd after applying to solved');
});

test('PLL parity alg preserves reduction and toggles permutation parity', () => {
  const s = applyMoves(solvedState(), PLL_PARITY_ALG);
  assert.ok(centersSolved(s), 'centers');
  assert.equal(pairedCount(wingsFromState(s)), 12, 'pairing');
  const info = analyze3x3(extract3x3(s));
  assert.equal(info.edgeParityOdd, false, 'flip parity unchanged');
  assert.equal(info.permParityMismatch, true, 'perm parity should mismatch');
});

test('extract3x3 of solved state is the solved facelet string', () => {
  const f = extract3x3(solvedState());
  assert.equal(f, 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB');
  const info = analyze3x3(f);
  assert.equal(info.ok, true);
});

test('extract3x3 tracks 3x3-like moves', () => {
  const s = applyMoves(solvedState(), "R U R' U'");
  const info = analyze3x3(extract3x3(s));
  assert.equal(info.ok, true);
});
