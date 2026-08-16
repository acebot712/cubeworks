// 3x3 stage: a reduced 4x4 (centers solved, edges paired) is mapped to a 3x3
// facelet string and solved with the Kociemba two-phase solver (cubejs).
// Parities that make the reduced state 3x3-invalid are fixed first:
//   - OLL parity: odd edge-flip sum  -> odd-wing-permutation alg
//   - PLL parity: edge/corner permutation parity mismatch -> 2R2-style alg
// Both algs are fixed sequences, verified by tests to preserve the reduction.
import Cube from 'cubejs';
import { FACES, FACE_INDEX } from '../cube/geometry.js';
import { parseMoves } from '../cube/notation.js';
import { applyMoves } from '../cube/state.js';
import { OLL_PARITY_ALG, PLL_PARITY_ALG } from './algs.js';

// map 3x3 (r,c in 0..2) to a 4x4 facelet (r,c in 0..3); middles use (1,1)-ish
const m3to4 = (v) => (v === 0 ? 0 : v === 1 ? 1 : 3);

export function extract3x3(state) {
  // Kociemba facelet order: U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9
  const order = ['U', 'R', 'F', 'D', 'L', 'B'];
  let out = '';
  for (const f of order) {
    const fi = FACE_INDEX[f];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        // careful: for edge facelets on a 4x4 the two candidate stickers are
        // identical because edges are paired; centers identical because solved
        const idx = fi * 16 + m3to4(r) * 4 + m3to4(c);
        out += FACES[state[idx]];
      }
    }
  }
  return out;
}

// --- 3x3 validity math on a facelet string -------------------------------
const EDGE_FACELETS = [
  [5, 10], [7, 19], [3, 37], [1, 46],       // UR UF UL UB
  [32, 16], [28, 25], [30, 43], [34, 52],   // DR DF DL DB
  [23, 12], [21, 41], [50, 39], [48, 14],   // FR FL BL BR
];
const EDGE_PIECES = ['UR', 'UF', 'UL', 'UB', 'DR', 'DF', 'DL', 'DB', 'FR', 'FL', 'BL', 'BR'];
const CORNER_FACELETS = [
  [8, 9, 20], [6, 18, 38], [0, 36, 47], [2, 45, 11],
  [29, 26, 15], [27, 44, 24], [33, 53, 42], [35, 17, 51],
];
const CORNER_PIECES = ['URF', 'UFL', 'ULB', 'UBR', 'DFR', 'DLF', 'DBL', 'DRB'];

export function analyze3x3(facelets) {
  // returns { ok, edgeParityOdd, permParityMismatch, error }
  const edges = [];
  for (const [a, b] of EDGE_FACELETS) {
    const pair = facelets[a] + facelets[b];
    let found = -1, flip = 0;
    for (let i = 0; i < 12; i++) {
      const p = EDGE_PIECES[i];
      if (pair === p) { found = i; flip = 0; break; }
      if (pair === p[1] + p[0]) { found = i; flip = 1; break; }
    }
    if (found < 0) return { ok: false, error: 'unrecognized edge ' + pair };
    edges.push({ piece: found, flip });
  }
  const corners = [];
  for (const [a, b, c] of CORNER_FACELETS) {
    const tri = facelets[a] + facelets[b] + facelets[c];
    let found = -1, twist = -1;
    for (let i = 0; i < 8; i++) {
      const p = CORNER_PIECES[i];
      for (let t = 0; t < 3; t++) {
        const rot = p[t % 3] + p[(1 + t) % 3] + p[(2 + t) % 3];
        if (tri === rot) { found = i; twist = t; break; }
      }
      if (found >= 0) break;
    }
    if (found < 0) return { ok: false, error: 'unrecognized corner ' + tri };
    corners.push({ piece: found, twist });
  }
  // piece sets must be complete
  if (new Set(edges.map((e) => e.piece)).size !== 12) return { ok: false, error: 'duplicate edges' };
  if (new Set(corners.map((c) => c.piece)).size !== 8) return { ok: false, error: 'duplicate corners' };

  const flipSum = edges.reduce((a, e) => a + e.flip, 0) % 2;
  const twistSum = corners.reduce((a, c) => a + c.twist, 0) % 3;
  if (twistSum !== 0) return { ok: false, error: 'corner twist parity' };

  const sign = (arr) => {
    const p = arr.slice();
    let s = 1;
    const seen = new Array(p.length).fill(false);
    for (let i = 0; i < p.length; i++) {
      if (seen[i]) continue;
      let len = 0, j = i;
      while (!seen[j]) { seen[j] = true; j = p[j]; len++; }
      if (len % 2 === 0) s = -s;
    }
    return s;
  };
  const edgeSign = sign(edges.map((e) => e.piece));
  const cornerSign = sign(corners.map((c) => c.piece));

  return {
    ok: flipSum === 0 && edgeSign === cornerSign,
    edgeParityOdd: flipSum === 1,
    permParityMismatch: edgeSign !== cornerSign,
  };
}

let solverReady = false;
export function initThreeSolver() {
  if (!solverReady) { Cube.initSolver(); solverReady = true; }
}

export function solveThreeStage(state96) {
  initThreeSolver();
  let state = state96;
  const parityMoves = [];

  let info = analyze3x3(extract3x3(state));
  if (info.error) throw new Error('reduced state invalid: ' + info.error);
  if (info.edgeParityOdd) {
    state = applyMoves(state, OLL_PARITY_ALG);
    parityMoves.push(...OLL_PARITY_ALG);
    info = analyze3x3(extract3x3(state));
    if (info.error) throw new Error('after OLL parity: ' + info.error);
  }
  if (info.permParityMismatch) {
    state = applyMoves(state, PLL_PARITY_ALG);
    parityMoves.push(...PLL_PARITY_ALG);
    info = analyze3x3(extract3x3(state));
    if (info.error) throw new Error('after PLL parity: ' + info.error);
  }
  if (!info.ok) throw new Error('reduced state still invalid after parity fixes');

  const facelets = extract3x3(state);
  if (facelets === 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB') {
    return { parityMoves, moves: [], state };
  }
  const cube = Cube.fromString(facelets);
  const sol = cube.solve(24);
  if (sol == null) throw new Error('Kociemba solver failed');
  const moves = parseMoves(sol);
  state = applyMoves(state, moves);
  return { parityMoves, moves, state };
}
