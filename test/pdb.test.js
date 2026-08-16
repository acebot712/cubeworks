import test from 'node:test';
import assert from 'node:assert/strict';
import { pdbIndex, pdbUnindex, PDB_SIZE, U_RANKS, D_RANKS } from '../src/solver/pdb.js';

// The full table takes ~7 minutes to build, so the suite verifies the INDEXING
// — which is what correctness of the stored distances rests on. A ranking that
// collides would silently corrupt every entry.

test('index space is exactly C(24,4) x C(20,4)', () => {
  assert.equal(U_RANKS, 10626);
  assert.equal(D_RANKS, 4845);
  assert.equal(PDB_SIZE, 10626 * 4845);
});

test('index and unindex round-trip across the whole range', () => {
  const buf = new Uint8Array(24);
  for (const idx of [0, 1, 2, 12345, 1_000_000, 25_000_000, PDB_SIZE - 1]) {
    pdbUnindex(idx, buf);
    assert.equal(pdbIndex(buf), idx, `round-trip failed at ${idx}`);
  }
});

test('indexing is injective on a large random sample', () => {
  const buf = new Uint8Array(24);
  const byArrangement = new Map();
  let s = 12345;
  for (let i = 0; i < 20000; i++) {
    s = (s * 1103515245 + 12345) % 2147483648;
    const idx = s % PDB_SIZE;
    pdbUnindex(idx, buf);
    const key = buf.join('');
    // Two DIFFERENT indices mapping to one arrangement would corrupt the table.
    // The same index appearing twice is just the sampler repeating itself —
    // with 20k draws from 51.5M, the birthday bound expects a handful.
    const prev = byArrangement.get(key);
    if (prev !== undefined) assert.equal(prev, idx, `indices ${prev} and ${idx} collide`);
    byArrangement.set(key, idx);
    assert.equal(pdbIndex(buf), idx);
  }
});

test('every unindexed arrangement has exactly 4 of each tracked face', () => {
  const buf = new Uint8Array(24);
  let s = 999;
  for (let i = 0; i < 2000; i++) {
    s = (s * 1103515245 + 12345) % 2147483648;
    pdbUnindex(s % PDB_SIZE, buf);
    let a = 0, b = 0;
    for (const v of buf) { if (v === 0) a++; else if (v === 3) b++; }
    assert.equal(a, 4);
    assert.equal(b, 4);
  }
});
