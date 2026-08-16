// Applying moves to a cube state. State = Uint8Array(96) of face codes.
import { FACES, FACE_INDEX } from './geometry.js';
import { PERMS } from './moves.js';
import { parseMoves } from './notation.js';

export function solvedState() {
  const s = new Uint8Array(96);
  for (let i = 0; i < 96; i++) s[i] = (i / 16) | 0;
  return s;
}

export function applyMove(state, token) {
  const perm = PERMS[token];
  if (!perm) throw new Error('unknown move: ' + token);
  const out = new Uint8Array(96);
  for (let i = 0; i < 96; i++) out[perm[i]] = state[i];
  return out;
}

export function applyMoves(state, moves) {
  let s = state;
  for (const m of parseMoves(moves)) s = applyMove(s, m);
  return s;
}

export function stateToString(state) {
  return Array.from(state).map((v) => FACES[v]).join('');
}

// One face's 16 stickers as face letters, row-major.
export function faceGrid(state, face) {
  const fi = FACE_INDEX[face];
  const out = [];
  for (let i = 0; i < 16; i++) out.push(FACES[state[fi * 16 + i]]);
  return out;
}
