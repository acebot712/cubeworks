// Move permutations, generated from 3D cubie geometry.
//
// Nothing here is hand-tabled: each quarter turn is derived by rotating every
// affected facelet's position and normal and looking up where it lands, so the
// whole move set is consistent by construction.
import { FACELET_POS, faceletAt } from './geometry.js';
import { MOVE_DEFS, MOVE_BASES } from './moveTable.js';

// Quarter-turn coordinate transforms (positive-face clockwise: R-like, U-like,
// F-like). Points use grid coords 0..3; normals use the same linear map on
// components.
function rotPoint(axis, p) {
  const [x, y, z] = p;
  if (axis === 'x') return [x, z, 3 - y];      // R-like: F->U
  if (axis === 'y') return [3 - z, y, x];      // U-like: F->L
  return [y, 3 - x, z];                        // F-like: U->R
}
function rotNormal(axis, n) {
  const [x, y, z] = n;
  if (axis === 'x') return [x, z, -y];
  if (axis === 'y') return [-z, y, x];
  return [y, -x, z];
}

// A base quarter move = axis + the set of layer coords along that axis.
// Returns perm where newState[perm[i]] = oldState[i].
function makeQuarterPerm(axis, layers) {
  const ax = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
  const layerSet = new Set(layers);
  const perm = new Int16Array(96);
  for (let i = 0; i < 96; i++) {
    const { p, n } = FACELET_POS[i];
    if (!layerSet.has(p[ax])) { perm[i] = i; continue; }
    const dest = faceletAt(rotPoint(axis, p), rotNormal(axis, n));
    if (dest === undefined) throw new Error('geometry bug');
    perm[i] = dest;
  }
  return perm;
}

function composePerm(a, b) { // apply a then b
  const out = new Int16Array(96);
  for (let i = 0; i < 96; i++) out[i] = b[a[i]];
  return out;
}

// PERMS[token] for token like "U", "U'", "U2", "u", "Rw'", "x2", ...
export const PERMS = {};
for (const name of MOVE_BASES) {
  const [axis, layers, times] = MOVE_DEFS[name];
  let q = makeQuarterPerm(axis, layers);
  if (times === 3) q = composePerm(composePerm(q, q), q); // D/L/B bases are CCW of axis
  const q2 = composePerm(q, q);
  PERMS[name] = q;
  PERMS[name + '2'] = q2;
  PERMS[name + "'"] = composePerm(q2, q);
}
