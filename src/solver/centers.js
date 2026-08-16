// Centers solver: brings all 24 center facelets to their home faces.
// Strategy: solve faces in order U, D, F, R, B (L completes automatically).
// At each step run an iterative-deepening search over center-only states for a
// short sequence that adds at least one correct center on the current face
// while keeping all previously completed faces intact. Correctness of every
// accepted sequence is guaranteed by simulation, not by case analysis.
import { CENTER_PERMS, centersFromState } from './tables.js';
import { applyMoves } from '../cube/state.js';
import { intPicker } from '../cube/rng.js';

const FACE_ORDER = [0, 3, 2, 1, 5]; // U, D, F, R, B (indices in URFDLB)

// Moves that can affect center arrangement (slices reposition between faces,
// outer turns rotate a face's own centers for setup).
const TOKENS = [];
for (const b of ['u', 'd', 'r', 'l', 'f', 'b', 'U', 'D', 'R', 'L', 'F', 'B']) {
  TOKENS.push(b, b + "'", b + '2');
}
const TOKEN_PERMS = TOKENS.map((t) => CENTER_PERMS[t]);
const TOKEN_BASE = TOKENS.map((t) => t.replace(/['2]/g, ''));
const AXIS = { u: 0, d: 0, U: 0, D: 0, r: 1, l: 1, R: 1, L: 1, f: 2, b: 2, F: 2, B: 2 };
const TOKEN_AXIS = TOKEN_BASE.map((b) => AXIS[b]);
const BASE_RANK = {};
['u', 'd', 'U', 'D', 'r', 'l', 'R', 'L', 'f', 'b', 'F', 'B'].forEach((b, i) => { BASE_RANK[b] = i; });
const TOKEN_RANK = TOKEN_BASE.map((b) => BASE_RANK[b]);

function countFace(centers, face) {
  let n = 0;
  for (let k = 0; k < 4; k++) if (centers[face * 4 + k] === face) n++;
  return n;
}

// Search for a sequence up to maxDepth that yields positive gain.
// gainFn(centers) -> number or -1 for constraint violation.
function search(centers, maxDepth, gainFn, fullScan) {
  const stack = [centers];
  for (let d = 1; d <= maxDepth; d++) stack.push(new Uint8Array(24));
  const seq = [];
  let best = null;
  let bestGain = 0;
  let scan = false;

  function dfs(depth, maxD) {
    const cur = stack[depth];
    for (let t = 0; t < TOKENS.length; t++) {
      if (depth > 0) {
        const prev = seq[depth - 1];
        if (TOKEN_BASE[t] === TOKEN_BASE[prev]) continue;
        // canonical order for commuting same-axis moves
        if (TOKEN_AXIS[t] === TOKEN_AXIS[prev] && TOKEN_RANK[t] < TOKEN_RANK[prev]) continue;
      }
      const perm = TOKEN_PERMS[t];
      const next = stack[depth + 1];
      for (let i = 0; i < 24; i++) next[perm[i]] = cur[i];
      seq[depth] = t;
      if (depth + 1 === maxD) {
        const g = gainFn(next);
        if (g > bestGain) {
          bestGain = g;
          best = seq.slice(0, depth + 1).map((x) => TOKENS[x]);
          if (!scan) return true;
        }
      } else if (dfs(depth + 1, maxD)) {
        return true;
      }
    }
    seq.length = depth;
    return false;
  }

  for (let maxD = 1; maxD <= maxDepth; maxD++) {
    scan = fullScan && maxD <= 3; // shallow depths: pick the best gain, not the first
    best = null; bestGain = 0; seq.length = 0;
    dfs(0, maxD);
    if (best) return best;
  }
  return null;
}

const rnd = intPicker(12345);

export function solveCenters(state96) {
  let state = state96;
  const solution = [];
  const done = [];

  for (const face of FACE_ORDER) {
    let guard = 0;
    while (countFace(centersFromState(state), face) < 4) {
      if (++guard > 60) throw new Error('centers solver stuck on face ' + face);
      const centers = centersFromState(state);
      const base = countFace(centers, face);
      const gainFn = (c) => {
        for (const f of done) if (countFace(c, f) < 4) return -1;
        return Math.max(0, countFace(c, face) - base);
      };
      let seq = search(centers, 4, gainFn, true);
      if (!seq) seq = search(centers, 5, gainFn, false);
      if (!seq) {
        // escape a local minimum: apply a random slice move that keeps
        // completed faces intact, then keep searching
        for (let tries = 0; tries < 40; tries++) {
          const tok = TOKENS[rnd(TOKENS.length)];
          const c2 = new Uint8Array(24);
          const perm = CENTER_PERMS[tok];
          for (let i = 0; i < 24; i++) c2[perm[i]] = centers[i];
          if (done.every((f) => countFace(c2, f) === 4)) { seq = [tok]; break; }
        }
        if (!seq) throw new Error('centers solver: no escape move');
      }
      state = applyMoves(state, seq);
      solution.push(...seq);
    }
    done.push(face);
  }
  return { moves: solution, state };
}
