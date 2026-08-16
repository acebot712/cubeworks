// Pattern database: exact solving distances for a relaxed sub-problem.
//
// This is the machinery behind both halves of "provably optimal where possible":
//
//   * A PDB entry is the EXACT number of moves to solve a simplified version of
//     the cube. Because the real cube is at least as hard as the simplification,
//     that number is a LOWER BOUND on the real distance — it can never
//     overestimate. A heuristic with that property is called admissible, and
//     IDA* driven by an admissible heuristic provably returns a shortest
//     solution. That is the whole basis of optimal solving.
//
//   * The same number, applied to a scanned cube, certifies the gap: if we find
//     a 94-move solution and the PDB proves no solution shorter than 38 exists,
//     the answer is "94, and provably within 56 of optimal". If the two numbers
//     ever meet, the solution is proven optimal for that configuration.
//
// The relaxation here tracks only WHERE the U-centres and D-centres sit,
// ignoring every other piece. That is C(24,4) x C(20,4) = 51,482,970 states —
// small enough to enumerate exhaustively by breadth-first search from solved,
// which is what makes the stored distances exact rather than estimated.
import { CENTER_PERMS, centersFromState } from './tables.js';

const N_SLOTS = 24;
const K = 4;

// --- combinatorial ranking: map a 4-subset of n slots to a dense index -------
const C = [];
for (let n = 0; n <= N_SLOTS; n++) {
  C.push([]);
  for (let k = 0; k <= K; k++) {
    C[n].push(k === 0 ? 1 : n === 0 ? 0 : C[n - 1][k - 1] + C[n - 1][k]);
  }
}

// rank a sorted 4-subset of [0, n) in colexicographic order
function rankSubset(sorted, k = K) {
  let r = 0;
  for (let i = 0; i < k; i++) r += C[sorted[i]][i + 1];
  return r;
}

export const U_RANKS = C[24][4];       // 10626
export const D_RANKS = C[20][4];       // 4845
export const PDB_SIZE = U_RANKS * D_RANKS;

// Index a centre arrangement by (positions of U-centres, positions of
// D-centres among the remaining slots).
export function pdbIndex(centers, faceA = 0, faceB = 3) {
  const a = [], rest = [];
  for (let i = 0; i < N_SLOTS; i++) {
    if (centers[i] === faceA) a.push(i); else rest.push(i);
  }
  if (a.length !== K) return -1;
  // positions of faceB expressed as offsets within `rest`
  const b = [];
  for (let i = 0; i < rest.length; i++) if (centers[rest[i]] === faceB) b.push(i);
  if (b.length !== K) return -1;
  return rankSubset(a) * D_RANKS + rankSubset(b);
}

// The moves that can rearrange centres.
export const PDB_TOKENS = (() => {
  const out = [];
  for (const b of ['u', 'd', 'r', 'l', 'f', 'b', 'U', 'D', 'R', 'L', 'F', 'B']) {
    out.push(b, `${b}'`, `${b}2`);
  }
  return out;
})();
const PDB_PERMS = PDB_TOKENS.map((t) => CENTER_PERMS[t]);

// unrank a colex-ranked k-subset back to its sorted elements
function unrankSubset(r, k, out) {
  for (let i = k - 1; i >= 0; i--) {
    let c = i;
    while (C[c + 1][i + 1] <= r) c++;
    out[i] = c;
    r -= C[c][i + 1];
  }
  return out;
}

const _a = new Uint8Array(K), _b = new Uint8Array(K), _rest = new Uint8Array(20);

// index -> centre arrangement, written into `out`
export function pdbUnindex(idx, out, faceA = 0, faceB = 3) {
  const rb = idx % D_RANKS, ra = (idx - rb) / D_RANKS;
  unrankSubset(ra, K, _a);
  unrankSubset(rb, K, _b);
  out.fill(9);
  for (let i = 0; i < K; i++) out[_a[i]] = faceA;
  let n = 0;
  for (let i = 0; i < N_SLOTS; i++) if (out[i] !== faceA) _rest[n++] = i;
  for (let i = 0; i < K; i++) out[_rest[_b[i]]] = faceB;
  return out;
}

// Build the table by BFS from solved. Operates on INDICES, reconstructing each
// arrangement on demand — storing 51M arrangements as objects exhausts the heap,
// while the same queue as packed 32-bit indices is ~200MB.
// -> Uint8Array(PDB_SIZE), 255 meaning unreachable
export function buildPdb({ onProgress } = {}) {
  const dist = new Uint8Array(PDB_SIZE).fill(255);
  const queue = new Int32Array(PDB_SIZE);

  const solved = new Uint8Array(N_SLOTS).fill(9);
  for (let i = 0; i < 4; i++) { solved[i] = 0; solved[12 + i] = 3; }
  const start = pdbIndex(solved);
  dist[start] = 0;
  queue[0] = start;

  let head = 0, tail = 1, depth = 0, levelEnd = 1;
  const cur = new Uint8Array(N_SLOTS), next = new Uint8Array(N_SLOTS);

  while (head < tail) {
    if (head === levelEnd) {
      depth++;
      levelEnd = tail;
      if (onProgress) onProgress({ depth, seen: tail, frontier: tail - head });
    }
    const idx = queue[head++];
    pdbUnindex(idx, cur);
    const d = dist[idx] + 1;
    for (let t = 0; t < PDB_PERMS.length; t++) {
      const p = PDB_PERMS[t];
      for (let i = 0; i < N_SLOTS; i++) next[p[i]] = cur[i];
      const ni = pdbIndex(next);
      if (ni < 0 || dist[ni] !== 255) continue;
      dist[ni] = d;
      queue[tail++] = ni;
    }
  }
  return { dist, maxDepth: depth, reached: tail };
}

// -> a PROVEN lower bound on the moves needed to solve this cube, or 0.
// Never overestimates: solving the whole cube necessarily solves its centres.
export function lowerBound(state96, dist) {
  const centers = centersFromState(state96);
  const idx = pdbIndex(centers);
  if (idx < 0) return 0;
  const d = dist[idx];
  return d === 255 ? 0 : d;
}
