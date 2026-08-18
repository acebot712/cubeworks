// The move vocabulary: which layers each token turns, and how far.
//
// Pure token algebra, no permutations, no state. Both the permutation
// generator (moves.js) and the notation utilities (notation.js) read this
// table, and neither needs the other.

// token base -> [axis, layer coords along that axis, quarter turns in the
// POSITIVE-axis direction]. D/L/B are the negative-facing faces, so their
// clockwise turn is 3 positive quarters.
export const MOVE_DEFS = {
  U: ['y', [3], 1], u: ['y', [2], 1], Uw: ['y', [2, 3], 1],
  D: ['y', [0], 3], d: ['y', [1], 3], Dw: ['y', [0, 1], 3],
  R: ['x', [3], 1], r: ['x', [2], 1], Rw: ['x', [2, 3], 1],
  L: ['x', [0], 3], l: ['x', [1], 3], Lw: ['x', [0, 1], 3],
  F: ['z', [3], 1], f: ['z', [2], 1], Fw: ['z', [2, 3], 1],
  B: ['z', [0], 3], b: ['z', [1], 3], Bw: ['z', [0, 1], 3],
  x: ['x', [0, 1, 2, 3], 1], y: ['y', [0, 1, 2, 3], 1], z: ['z', [0, 1, 2, 3], 1],
};

export const MOVE_BASES = Object.keys(MOVE_DEFS);

export const tokenBase = (token) => token.replace(/['2]/g, '');

// Quarter turns a suffix means, in the token's OWN direction.
const suffixTurns = (token) => (token.endsWith('2') ? 2 : token.endsWith("'") ? 3 : 1);

// Net quarter turns in the positive-axis direction.
export function tokenTurns(token, base) {
  const [, , times] = MOVE_DEFS[base];
  return (suffixTurns(token) * times) % 4;
}

// The inverse of tokenTurns: rebuild a token from positive-axis quarters.
// Returns null for the identity.
export function tokenFromTurns(base, turns) {
  const [, , times] = MOVE_DEFS[base];
  const own = times === 3 ? (4 - turns) % 4 : turns;
  if (own === 0) return null;
  if (own === 1) return base;
  if (own === 2) return base + '2';
  return base + "'";
}

export const sameAxis = (a, b) => MOVE_DEFS[a][0] === MOVE_DEFS[b][0];

export function layersOverlap(a, b) {
  const layersA = new Set(MOVE_DEFS[a][1]);
  return MOVE_DEFS[b][1].some((l) => layersA.has(l));
}

// token -> {axis, layers, turns}, turns in positive-axis quarters and taking
// the shortest visual arc. Used by the 3D view to animate layer rotations.
export function moveInfo(token) {
  const base = tokenBase(token);
  const def = MOVE_DEFS[base];
  if (!def) return null;
  const [axis, layers] = def;
  let turns = tokenTurns(token, base);
  if (turns === 3) turns = -1;
  return { axis, layers, turns };
}
