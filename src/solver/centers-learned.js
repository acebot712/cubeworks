// Learned centres solver: beam search guided by a trained cost-to-go network.
//
// The shipped greedy solver fixes centres face by face, accepting any sequence
// that places at least one piece. Measured on real scrambles that costs ~37
// moves. This one asks a network "how far from solved is this arrangement?" and
// keeps the most promising states — measured ~15 moves for the same states.
//
// The network was trained by Deep Approximate Value Iteration (see
// mlx-solver/train_centers.py): scramble k moves from solved, and regress
// J(s) towards min over successors of 1 + J(s'), bootstrapping from a
// periodically-frozen copy of itself. Training data is free because scrambling
// generates labelled states endlessly — no dataset to collect.
//
// What this is NOT: proven optimal. A learned heuristic is not admissible, so
// nothing here certifies minimality. The information-theoretic floor for this
// sub-problem is ~10 moves, so ~15 is close, but "close" is all it is.
import { SHAPES, WEIGHTS_B64 } from './centers-net.js';
import { CENTER_PERMS, centersFromState } from './tables.js';
import { applyMoves } from '../cube/state.js';

const N_SLOTS = 24;
const N_FACES = 6;
const IN = N_SLOTS * N_FACES;

export const TOKENS = (() => {
  const out = [];
  for (const b of ['u', 'd', 'r', 'l', 'f', 'b', 'U', 'D', 'R', 'L', 'F', 'B']) {
    out.push(b, `${b}'`, `${b}2`);
  }
  return out;
})();
const PERMS = TOKENS.map((t) => CENTER_PERMS[t]);
const N_MOVES = TOKENS.length;

// --- weights ---------------------------------------------------------------
let NET = null;
function net() {
  if (NET) return NET;
  const bin = typeof atob === 'function'
    ? Uint8Array.from(atob(WEIGHTS_B64), (c) => c.charCodeAt(0))
    : Uint8Array.from(Buffer.from(WEIGHTS_B64, 'base64'));
  const all = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);
  const layers = [];
  let off = 0;
  for (const [outDim, inDim] of SHAPES) {
    const w = all.subarray(off, off + outDim * inDim); off += outDim * inDim;
    const b = all.subarray(off, off + outDim); off += outDim;
    layers.push({ w, b, outDim, inDim });
  }
  NET = layers;
  return NET;
}

// Forward pass over a batch. Rows of `states` are 24 face ids; the encoding is
// one-hot per slot, so the first matrix multiply only ever touches 24 of 144
// inputs — worth exploiting since this runs thousands of times per solve.
function evaluate(states, count) {
  const L = net();
  const first = L[0];
  let cur = new Float32Array(count * first.outDim);
  for (let s = 0; s < count; s++) {
    const base = s * N_SLOTS;
    const outBase = s * first.outDim;
    for (let o = 0; o < first.outDim; o++) cur[outBase + o] = first.b[o];
    for (let slot = 0; slot < N_SLOTS; slot++) {
      const inIdx = slot * N_FACES + states[base + slot];
      const row = inIdx;                    // column of W, laid out (out, in)
      for (let o = 0; o < first.outDim; o++) {
        cur[outBase + o] += first.w[o * first.inDim + row];
      }
    }
    for (let o = 0; o < first.outDim; o++) if (cur[outBase + o] < 0) cur[outBase + o] = 0;
  }
  let dim = first.outDim;
  for (let li = 1; li < L.length; li++) {
    const { w, b, outDim, inDim } = L[li];
    const next = new Float32Array(count * outDim);
    for (let s = 0; s < count; s++) {
      const ib = s * inDim, ob = s * outDim;
      for (let o = 0; o < outDim; o++) {
        let acc = b[o];
        const wr = o * inDim;
        for (let i = 0; i < inDim; i++) acc += w[wr + i] * cur[ib + i];
        next[ob + o] = li === L.length - 1 ? acc : (acc < 0 ? 0 : acc);
      }
    }
    cur = next; dim = outDim;
  }
  return cur;   // (count,) when the head has one output
}

const isSolved = (arr, off) => {
  for (let i = 0; i < N_SLOTS; i++) if (arr[off + i] !== ((i / 4) | 0)) return false;
  return true;
};

// -> { moves, nodes } or null
export function solveCentersLearned(state96, { width = 100, maxDepth = 40 } = {}) {
  const start = centersFromState(state96);
  if (isSolved(start, 0)) return { moves: [], nodes: 0 };

  let beam = Uint8Array.from(start);
  let beamCount = 1;
  let paths = [[]];
  let nodes = 0;

  for (let depth = 0; depth < maxDepth; depth++) {
    const kidCount = beamCount * N_MOVES;
    const kids = new Uint8Array(kidCount * N_SLOTS);
    for (let s = 0; s < beamCount; s++) {
      const sb = s * N_SLOTS;
      for (let m = 0; m < N_MOVES; m++) {
        const p = PERMS[m], kb = (s * N_MOVES + m) * N_SLOTS;
        for (let i = 0; i < N_SLOTS; i++) kids[kb + p[i]] = beam[sb + i];
      }
    }
    nodes += kidCount;

    for (let k = 0; k < kidCount; k++) {
      if (isSolved(kids, k * N_SLOTS)) {
        return { moves: [...paths[(k / N_MOVES) | 0], TOKENS[k % N_MOVES]], nodes };
      }
    }

    const j = evaluate(kids, kidCount);
    const order = Array.from({ length: kidCount }, (_, i) => i);
    order.sort((a, b) => j[a] - j[b]);

    // de-duplicate: the same arrangement reached twice wastes beam width
    const seen = new Set();
    const keep = [];
    for (const i of order) {
      const kb = i * N_SLOTS;
      let key = '';
      for (let t = 0; t < N_SLOTS; t++) key += kids[kb + t];
      if (seen.has(key)) continue;
      seen.add(key);
      keep.push(i);
      if (keep.length >= width) break;
    }

    const nextBeam = new Uint8Array(keep.length * N_SLOTS);
    const nextPaths = [];
    keep.forEach((i, n) => {
      nextBeam.set(kids.subarray(i * N_SLOTS, i * N_SLOTS + N_SLOTS), n * N_SLOTS);
      nextPaths.push([...paths[(i / N_MOVES) | 0], TOKENS[i % N_MOVES]]);
    });
    beam = nextBeam; beamCount = keep.length; paths = nextPaths;
    if (!beamCount) return null;
  }
  return null;
}

// Same contract as solveCenters, so it can be swapped in directly.
//
// Escalates the beam rather than giving up: a narrow beam solves most states in
// well under a second, and the occasional hard one is worth a wider retry. The
// caller keeps its greedy result if every width fails, so this can only ever
// help.
export function solveCentersLearnedVerified(state96, { widths = [100, 300, 800], maxDepth = 40 } = {}) {
  for (const width of widths) {
    const r = solveCentersLearned(state96, { width, maxDepth });
    if (!r) continue;
    const state = applyMoves(state96, r.moves);
    // never return an unverified solution, however confident the network is
    if (!isSolved(centersFromState(state), 0)) continue;
    return { moves: r.moves, state, nodes: r.nodes, width };
  }
  return null;
}
