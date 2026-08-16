// Edge pairing: pairs all 12 edges of a centers-solved 4x4 (reduction step 2).
//
// Core structure: slice-flip-slice macros of the form  S1 · s · S2 · s'
// where s is an inner slice (u/u'/d/d'/u2/d2) and S1/S2 are short outer-move
// sequences found by search. Outer moves move edge slots as units (they can
// never pair or unpair anything) and only rotate their own face's centers.
//   - S1 (wings only): after S1·s at least one slot is paired that was not
//     paired before (slices may temporarily break other pairs; s' restores
//     any the S2 stage leaves in place).
//   - S2 (wings + centers): after S1·s·S2·s' the total pair count has grown
//     and every center is back on its home face.
// Every accepted macro is re-verified on the full 96-facelet state.
import {
  WING_MOVES, CENTER_PERMS, wingsFromState, centersFromState, pairedCount, SLOT_WINGS,
  OUTER_TOKENS, applyWingMove, applyCenterMove,
} from './tables.js';
import { applyMoves } from '../cube/state.js';
import { EDGE_SLOTS } from '../cube/pieces.js';
import { intPicker } from '../cube/rng.js';
import { tryL2ETable } from './l2e.js';
import { OLL_PARITY_ALG, FLIP_FR_ALG } from './algs.js';

const OUTER = OUTER_TOKENS;
const OUTER_WINGS = OUTER.map((t) => WING_MOVES[t]);
const OUTER_CENTERS = OUTER.map((t) => CENTER_PERMS[t]);
const OUTER_BASE = OUTER.map((t) => t[0]);
const AXIS = { U: 0, D: 0, R: 1, L: 1, F: 2, B: 2 };
const OUTER_AXIS = OUTER_BASE.map((b) => AXIS[b]);
const BASE_RANK = { U: 0, D: 1, R: 2, L: 3, F: 4, B: 5 };
const OUTER_RANK = OUTER_BASE.map((b) => BASE_RANK[b]);

const SLICES = ['u', "u'", 'd', "d'", 'u2', 'd2'];
const SLICE_INV = { u: "u'", "u'": 'u', d: "d'", "d'": 'd', u2: 'u2', d2: 'd2' };

function centersSolved(c) {
  for (let i = 0; i < 24; i++) if (c[i] !== ((i / 4) | 0)) return false;
  return true;
}
function pairMask(w) {
  let m = 0;
  for (let s = 0; s < 12; s++) {
    const [a, b] = SLOT_WINGS[s];
    if (w[a] === w[b]) m |= 1 << s;
  }
  return m;
}
function bitCount(m) {
  let n = 0;
  while (m) { m &= m - 1; n++; }
  return n;
}

// Enumerate outer sequences up to maxDepth (DFS with same-face/commute
// pruning); visit(wings, centers, seq) may return true to stop everything.
function enumerateOuter(wings0, centers0, maxDepth, visit, trackCenters, budget = Infinity) {
  const wStack = [wings0];
  const cStack = [centers0];
  for (let d = 1; d <= maxDepth; d++) { wStack.push(new Uint8Array(24)); cStack.push(new Uint8Array(24)); }
  const seq = [];
  let stopped = false;
  let nodes = 0;

  function dfs(depth, maxD) {
    if (stopped) return;
    for (let t = 0; t < OUTER.length; t++) {
      if (depth > 0) {
        const prev = seq[depth - 1];
        if (OUTER_BASE[t] === OUTER_BASE[prev]) continue;
        if (OUTER_AXIS[t] === OUTER_AXIS[prev] && OUTER_RANK[t] < OUTER_RANK[prev]) continue;
      }
      if (++nodes > budget) { stopped = true; return; }
      applyWingMove(wStack[depth + 1], wStack[depth], OUTER_WINGS[t]);
      if (trackCenters) applyCenterMove(cStack[depth + 1], cStack[depth], OUTER_CENTERS[t]);
      seq.push(t);
      if (visit(wStack[depth + 1], trackCenters ? cStack[depth + 1] : null, seq)) { stopped = true; return; }
      if (depth + 1 < maxD) dfs(depth + 1, maxD);
      seq.pop();
      if (stopped) return;
    }
  }

  if (visit(wings0, trackCenters ? centers0 : null, [])) return;
  if (maxDepth > 0) dfs(0, maxDepth);
}

function applySeq(wings, centers, tokens) {
  let w = Uint8Array.from(wings);
  let c = Uint8Array.from(centers);
  const wT = new Uint8Array(24), cT = new Uint8Array(24);
  for (const tok of tokens) {
    applyWingMove(wT, w, WING_MOVES[tok]); w.set(wT);
    applyCenterMove(cT, c, CENTER_PERMS[tok]); c.set(cT);
  }
  return { w, c };
}

// One slice-flip-slice macro. Returns token list or null.
function findMacro(wings, centers, s1MaxDepth, s2MaxDepth, maxCandidates) {
  const base = pairedCount(wings);
  const baseMask = pairMask(wings);
  const wTmp = new Uint8Array(24);
  const wBack = new Uint8Array(24);
  const cBack = new Uint8Array(24);

  for (const slice of SLICES) {
    const sliceW = WING_MOVES[slice];
    const inv = SLICE_INV[slice];
    const invW = WING_MOVES[inv];
    const invC = CENTER_PERMS[inv];

    // S1 candidates: after S1 · s some slot is newly paired
    const candidates = [];
    enumerateOuter(wings, centers, s1MaxDepth, (w, c, seq) => {
      applyWingMove(wTmp, w, sliceW);
      const mask = pairMask(wTmp);
      const fresh = mask & ~baseMask;
      if (fresh) {
        candidates.push({
          seq: seq.slice(),
          score: bitCount(fresh) * 16 + bitCount(mask) - seq.length,
        });
      }
      return candidates.length >= 800;
    }, false);
    candidates.sort((a, b) => b.score - a.score);

    for (const cand of candidates.slice(0, maxCandidates)) {
      const mid = applySeq(wings, centers, [...cand.seq.map((t) => OUTER[t]), slice]);
      let found = null;
      enumerateOuter(mid.w, mid.c, s2MaxDepth, (w, c, seq) => {
        applyWingMove(wBack, w, invW);
        if (pairedCount(wBack) <= base) return false;
        applyCenterMove(cBack, c, invC);
        if (!centersSolved(cBack)) return false;
        found = seq.slice();
        return true;
      }, true, 400000);

      if (found) {
        return [
          ...cand.seq.map((t) => OUTER[t]),
          slice,
          ...found.map((t) => OUTER[t]),
          inv,
        ];
      }
    }
  }
  return null;
}

const rnd = intPicker(987654);

const FR_INDEX = EDGE_SLOTS.findIndex((e) => e.name === 'FR');

// Short outer setup that brings some unpaired slot to FR, then flips it.
function findFlipEscape(wings, centers) {
  const [fa, fb] = SLOT_WINGS[FR_INDEX];
  let setup = null;
  enumerateOuter(wings, centers, 3, (w, c, seq) => {
    if (w[fa] !== w[fb]) { setup = seq.slice(); return true; }
    return false;
  }, false);
  if (!setup) return null;
  return [...setup.map((t) => OUTER[t]), ...FLIP_FR_ALG];
}

// A centers-preserving random shake: s X Y s' verified by simulation.
function findPerturb(wings, centers) {
  for (let tries = 0; tries < 200; tries++) {
    const s = SLICES[rnd(SLICES.length)];
    const seq = [s, OUTER[rnd(OUTER.length)], OUTER[rnd(OUTER.length)], SLICE_INV[s]];
    const { c } = applySeq(wings, centers, seq);
    if (centersSolved(c)) return seq;
  }
  throw new Error('edge pairing: no valid perturbation found');
}

function findPairingMacro(wings, centers, thorough) {
  let macro = findMacro(wings, centers, 3, 4, 40);
  if (!macro && thorough) macro = findMacro(wings, centers, 4, 5, 15);
  return macro;
}

export function solveEdges(state96) {
  let state = state96;
  const solution = [];
  let guard = 0;
  let flipsSinceProgress = 0;
  let parityUsed = false;

  while (true) {
    const wings = wingsFromState(state);
    const centers = centersFromState(state);
    if (!centersSolved(centers)) throw new Error('edge stage: centers not solved');
    const paired = pairedCount(wings);
    if (paired === 12) break;
    if (++guard > 80) throw new Error('edge pairing stuck at ' + paired + ' pairs');

    let macro = findPairingMacro(wings, centers, false);
    let neutral = false;
    if (!macro) macro = tryL2ETable(wings);
    if (!macro && !parityUsed) {
      macro = OLL_PARITY_ALG.slice();
      neutral = true;
      parityUsed = true;
    } else if (!macro && flipsSinceProgress < 2) {
      macro = findFlipEscape(wings, centers);
      if (macro) { neutral = true; flipsSinceProgress++; }
    }
    if (!macro) macro = findPairingMacro(wings, centers, true);
    if (!macro) { macro = findPerturb(wings, centers); neutral = true; flipsSinceProgress = 0; }

    const next = applyMoves(state, macro);
    const nw = wingsFromState(next);
    const nc = centersFromState(next);
    if (!centersSolved(nc)) throw new Error('macro broke centers: ' + macro.join(' '));
    if (!neutral && pairedCount(nw) <= paired) throw new Error('macro made no progress: ' + macro.join(' '));
    if (pairedCount(nw) > paired) { flipsSinceProgress = 0; parityUsed = false; }
    state = next;
    solution.push(...macro);
  }
  return { moves: solution, state };
}
