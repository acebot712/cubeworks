// Which facelets belong to which physical piece.
//
// Grouping is by cubie position, so these fall straight out of the geometry:
// a cubie touching three outer faces is a corner, two is an edge wing, one is a
// centre. Validation and the solver's piece tables are both built on this.
import { FACES, FACE_INDEX, FACELET_POS, exteriorCount } from './geometry.js';

// The four centre facelets of each face (r, c in {1,2}).
export const CENTER_IDX = {};
for (const f of FACES) {
  const fi = FACE_INDEX[f];
  CENTER_IDX[f] = [fi * 16 + 5, fi * 16 + 6, fi * 16 + 9, fi * 16 + 10];
}

const cubieKey = (p) => p.join(',');

// Group facelets by the cubie they sit on, keeping only cubies with `exterior`
// outer faces.
function cubiesWith(exterior) {
  const byCubie = new Map();
  for (let i = 0; i < 96; i++) {
    const { p } = FACELET_POS[i];
    if (exteriorCount(p) !== exterior) continue;
    const k = cubieKey(p);
    if (!byCubie.has(k)) byCubie.set(k, []);
    byCubie.get(k).push(i);
  }
  return byCubie;
}

// The 24 edge wings, each two facelets on two outer faces.
export const WINGS = []; // [{cubie:[x,y,z], stickers:[i,j]}]
for (const [k, stickers] of cubiesWith(2)) {
  if (stickers.length === 2) {
    WINGS.push({ cubie: k.split(',').map(Number), stickers: stickers.sort((a, b) => a - b) });
  }
}

// The 12 edge slots: the two wings of a slot share both outer faces and differ
// only in the inner coordinate, so keying on the outer coords groups them.
export const EDGE_SLOTS = []; // [{name, key, wings:[wingIdxA, wingIdxB]}]
{
  const outerOnly = (v) => (v === 0 || v === 3 ? v : '*');
  const byEdge = new Map();
  WINGS.forEach((w, wi) => {
    const key = w.cubie.map(outerOnly).join(',');
    if (!byEdge.has(key)) byEdge.set(key, []);
    byEdge.get(key).push(wi);
  });
  for (const [key, wings] of byEdge) {
    const name = WINGS[wings[0]].stickers.map((s) => FACELET_POS[s].face).sort().join('');
    EDGE_SLOTS.push({ name, key, wings });
  }
}

// The 8 corners, three facelets each.
export const CORNERS = []; // [{cubie:[x,y,z], stickers:[i,j,k]}]
for (const [k, stickers] of cubiesWith(3)) {
  CORNERS.push({ cubie: k.split(',').map(Number), stickers });
}
