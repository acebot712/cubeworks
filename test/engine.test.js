import test from 'node:test';
import assert from 'node:assert';
import { FACES } from '../src/cube/geometry.js';
import { PERMS } from '../src/cube/moves.js';
import { invertMoves, simplifyMoves, toUserMoves } from '../src/cube/notation.js';
import { solvedState, applyMoves, applyMove, faceGrid, stateToString } from '../src/cube/state.js';
import { WINGS, EDGE_SLOTS, CORNERS } from '../src/cube/pieces.js';

test('solved state face grids are uniform', () => {
  const s = solvedState();
  for (const f of FACES) {
    assert.ok(faceGrid(s, f).every((v) => v === f), 'face ' + f);
  }
});

test('every move^4 (or ^2 for halves) is identity', () => {
  const s = solvedState();
  for (const tok of Object.keys(PERMS)) {
    const n = tok.endsWith('2') ? 2 : 4;
    let t = s;
    for (let i = 0; i < n; i++) t = applyMove(t, tok);
    assert.equal(stateToString(t), stateToString(s), tok);
  }
});

test("move then inverse is identity", () => {
  const s = solvedState();
  for (const tok of Object.keys(PERMS)) {
    const t = applyMoves(applyMove(s, tok), invertMoves([tok]));
    assert.equal(stateToString(t), stateToString(s), tok);
  }
});

test('U cycles side top rows F->L->B->R', () => {
  const s = applyMove(solvedState(), 'U');
  // F top row shows what was on R, L shows old F, B shows old L, R shows old B
  assert.deepEqual(faceGrid(s, 'F').slice(0, 4), ['R', 'R', 'R', 'R']);
  assert.deepEqual(faceGrid(s, 'L').slice(0, 4), ['F', 'F', 'F', 'F']);
  assert.deepEqual(faceGrid(s, 'B').slice(0, 4), ['L', 'L', 'L', 'L']);
  assert.deepEqual(faceGrid(s, 'R').slice(0, 4), ['B', 'B', 'B', 'B']);
  // U face itself is still uniform, other rows untouched
  assert.ok(faceGrid(s, 'U').every((v) => v === 'U'));
  assert.deepEqual(faceGrid(s, 'F').slice(4), Array(12).fill('F'));
});

test('R brings F right column to U right column', () => {
  const s = applyMove(solvedState(), 'R');
  const u = faceGrid(s, 'U');
  // U right column (c=3) shows old F
  assert.deepEqual([u[3], u[7], u[11], u[15]], ['F', 'F', 'F', 'F']);
  const f = faceGrid(s, 'F');
  assert.deepEqual([f[3], f[7], f[11], f[15]], ['D', 'D', 'D', 'D']);
  const b = faceGrid(s, 'B');
  // B col 0 is at the R side; old U right column goes to B col 0
  assert.deepEqual([b[0], b[4], b[8], b[12]], ['U', 'U', 'U', 'U']);
});

test('inner slice u only moves band rows, not U/D centers', () => {
  const s = applyMove(solvedState(), 'u');
  assert.ok(faceGrid(s, 'U').every((v) => v === 'U'));
  assert.ok(faceGrid(s, 'D').every((v) => v === 'D'));
  const f = faceGrid(s, 'F');
  assert.deepEqual(f.slice(4, 8), ['R', 'R', 'R', 'R']); // row 1 from R (u moves like U)
  assert.deepEqual(f.slice(0, 4), ['F', 'F', 'F', 'F']);
  assert.deepEqual(f.slice(8), Array(8).fill('F'));
});

test('rotations: x maps F face to U slot', () => {
  const s = applyMove(solvedState(), 'x');
  assert.ok(faceGrid(s, 'U').every((v) => v === 'F'));
  assert.ok(faceGrid(s, 'F').every((v) => v === 'D'));
  assert.ok(faceGrid(s, 'B').every((v) => v === 'U'));
  const s2 = applyMove(solvedState(), 'y');
  assert.ok(faceGrid(s2, 'F').every((v) => v === 'R'));
  const s3 = applyMove(solvedState(), 'z');
  assert.ok(faceGrid(s3, 'U').every((v) => v === 'L'));
});

test('Uw == u then U (composition consistency)', () => {
  const a = applyMove(solvedState(), 'Uw');
  const b = applyMoves(solvedState(), 'u U');
  assert.equal(stateToString(a), stateToString(b));
});

test('sexy move has order 6', () => {
  let s = solvedState();
  for (let i = 0; i < 6; i++) s = applyMoves(s, "R U R' U'");
  assert.equal(stateToString(s), stateToString(solvedState()));
});

test('piece bookkeeping: 24 wings, 12 edge slots, 8 corners', () => {
  assert.equal(WINGS.length, 24);
  assert.equal(EDGE_SLOTS.length, 12);
  assert.ok(EDGE_SLOTS.every((e) => e.wings.length === 2));
  assert.equal(CORNERS.length, 8);
  assert.ok(CORNERS.every((c) => c.stickers.length === 3));
});

test('simplify merges and cancels', () => {
  assert.deepEqual(simplifyMoves("U U"), ['U2']);
  assert.deepEqual(simplifyMoves("U U'"), []);
  assert.deepEqual(simplifyMoves("R L R'"), ['L']);
  assert.deepEqual(simplifyMoves("U2 U2 F"), ['F']);
  assert.deepEqual(simplifyMoves("D D D"), ["D'"]);
});

test('toUserMoves expands inner slices and stays equivalent', () => {
  const seq = "r U2 r' d B l2";
  const user = toUserMoves(seq);
  assert.ok(user.every((m) => /^[UDFBRLxyz]w?['2]?$/.test(m)), user.join(' '));
  const a = applyMoves(solvedState(), seq);
  const b = applyMoves(solvedState(), user);
  assert.equal(stateToString(a), stateToString(b));
});
