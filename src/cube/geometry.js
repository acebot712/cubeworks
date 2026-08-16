// Where every sticker is in space.
//
// State is a Uint8Array(96) of face codes; facelet index = face * 16 + row * 4 +
// col, faces in Kociemba order URFDLB. This module is the bridge between that
// flat index and 3D: each facelet has a cubie position p and an outward normal
// n, which is what lets moves be derived from geometry instead of hand-tabled.
//
// Axes: x toward R, y toward U, z toward F. Cubie coords in {0..3}.
// Face viewing conventions (standard Kociemba):
//   U viewed from above, row 0 adjacent to B;  D viewed from below, row 0 adjacent to F
//   F/B/R/L viewed from outside with U on top; B col 0 at the R side, L col 0 at the B side.

export const FACES = ['U', 'R', 'F', 'D', 'L', 'B'];
export const FACE_INDEX = { U: 0, R: 1, F: 2, D: 3, L: 4, B: 5 };
const CUBE_N = 4;

function faceletToPos(f, r, c) {
  switch (f) {
    case 'U': return { p: [c, 3, r], n: [0, 1, 0] };
    case 'D': return { p: [c, 0, 3 - r], n: [0, -1, 0] };
    case 'F': return { p: [c, 3 - r, 3], n: [0, 0, 1] };
    case 'B': return { p: [3 - c, 3 - r, 0], n: [0, 0, -1] };
    case 'R': return { p: [3, 3 - r, 3 - c], n: [1, 0, 0] };
    case 'L': return { p: [0, 3 - r, c], n: [-1, 0, 0] };
    default: throw new Error('bad face ' + f);
  }
}

const posKey = (p, n) => p.join(',') + '|' + n.join(',');

const byPos = new Map(); // "x,y,z|nx,ny,nz" -> facelet index
export const FACELET_POS = []; // facelet index -> {face, r, c, p, n}
for (let fi = 0; fi < 6; fi++) {
  for (let r = 0; r < CUBE_N; r++) {
    for (let c = 0; c < CUBE_N; c++) {
      const idx = fi * 16 + r * 4 + c;
      const { p, n } = faceletToPos(FACES[fi], r, c);
      FACELET_POS[idx] = { face: FACES[fi], r, c, p, n };
      byPos.set(posKey(p, n), idx);
    }
  }
}

// The facelet at cubie position p facing along normal n, or undefined.
export const faceletAt = (p, n) => byPos.get(posKey(p, n));

// How many outer faces a cubie at p touches: 3 = corner, 2 = wing, 1 = centre.
export const exteriorCount = (p) =>
  (p[0] === 0 || p[0] === 3 ? 1 : 0) + (p[1] === 0 || p[1] === 3 ? 1 : 0) + (p[2] === 0 || p[2] === 3 ? 1 : 0);
