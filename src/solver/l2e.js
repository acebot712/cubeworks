// Last-two-edges endgame: when exactly two edge slots remain unpaired, the
// generic macro search can hit configurations that need long fixed algorithms
// (the classic L2E cases). We normalize the two unpaired slots to FR and FL
// with a short outer setup, classify the four wing displays into a
// color-agnostic class key, and look up a precomputed solving sequence.
// The table was generated offline by deep search (see scratch-l2e-table.mjs)
// and every application is re-verified by simulation at runtime.
import { EDGE_SLOTS } from '../cube/pieces.js';
import { WING_MOVES, SLOT_WINGS, OUTER_TOKENS, applyWingMove, flipWing } from './tables.js';
// classKey -> solving tokens, generated offline by deep search
// (see scratch-l2e-table.mjs); verified by tests and re-verified at runtime.
import L2E_TABLE from './l2e-data.js';

const FR = EDGE_SLOTS.findIndex((e) => e.name === 'FR');
const FL = EDGE_SLOTS.findIndex((e) => e.name === 'FL');

function l2eClassKey(w) {
  const vals = [...SLOT_WINGS[FR], ...SLOT_WINGS[FL]].map((i) => w[i]);
  const symbols = new Map();
  let next = 0;
  return vals.map((v) => {
    const baseV = Math.min(v, flipWing(v));
    if (!symbols.has(baseV)) symbols.set(baseV, next++);
    return symbols.get(baseV) + (v !== baseV ? 'f' : 'n');
  }).join('.');
}

// outer setup (<=5) bringing both unpaired slots to FR and FL
function findL2ESetup(wings) {
  const maxDepth = 5;
  const stack = [wings];
  for (let d = 1; d <= maxDepth; d++) stack.push(new Uint8Array(24));
  const seq = [];
  const [fra, frb] = SLOT_WINGS[FR];
  const [fla, flb] = SLOT_WINGS[FL];
  const good = (w) => w[fra] !== w[frb] && w[fla] !== w[flb];
  let found = null;
  function dfs(depth, maxD) {
    for (let t = 0; t < OUTER_TOKENS.length; t++) {
      if (depth > 0 && OUTER_TOKENS[t][0] === OUTER_TOKENS[seq[depth - 1]][0]) continue;
      applyWingMove(stack[depth + 1], stack[depth], WING_MOVES[OUTER_TOKENS[t]]);
      seq[depth] = t;
      if (good(stack[depth + 1])) { found = seq.slice(0, depth + 1).map((x) => OUTER_TOKENS[x]); return true; }
      if (depth + 1 < maxD && dfs(depth + 1, maxD)) return true;
    }
    return false;
  }
  if (good(wings)) return [];
  for (let maxD = 1; maxD <= maxDepth && !found; maxD++) dfs(0, maxD);
  return found;
}

function wingsAfter(wings, tokens) {
  let w = Uint8Array.from(wings);
  const t = new Uint8Array(24);
  for (const tok of tokens) {
    const mv = WING_MOVES[tok];
    if (mv) { applyWingMove(t, w, mv); w.set(t); }
  }
  return w;
}

// If exactly two slots are unpaired and the class is known, return a full
// verified sequence (setup + alg); otherwise null.
export function tryL2ETable(wings) {
  let unpaired = 0;
  for (const [a, b] of SLOT_WINGS) if (wings[a] !== wings[b]) unpaired++;
  if (unpaired !== 2) return null;
  const setup = findL2ESetup(wings);
  if (!setup) return null;
  const norm = wingsAfter(wings, setup);
  const alg = L2E_TABLE[l2eClassKey(norm)];
  if (!alg) return null;
  const final = wingsAfter(norm, alg);
  for (const [a, b] of SLOT_WINGS) if (final[a] !== final[b]) return null;
  return [...setup, ...alg];
}
