// Structural error correction for a scanned cube state.
//
// This is the Reed-Solomon half of the scanner: a legal 4x4 state is enormously
// redundant (16 stickers per color, 4 centers per color, 24 valid wing pairs, 8
// distinct corners), so most misreads are not just detectable but *correctable*.
// Two failure modes dominate in practice:
//   A. a whole face captured at the wrong in-plane rotation, or R/L yawed the
//      other way, a 4^6 x 2 search, checked against validateState
//   B. a handful of individual stickers misclassified (red/orange under warm
//      light), a small guided beam search over low-confidence stickers
//
// Pure module. Capture maps are passed in so cube/ never imports from scan/.
import { validateState } from './validate.js';
import { stateFromColors } from './colorState.js';
import { COLOR_KEYS, COLOR_OF_FACE, NEAR } from '../state/colors.js';

// How many individual sticker substitutions phase B is allowed to try, and how
// wide its beam is. Beyond this the scan is wrong enough that re-scanning a
// face beats guessing.
const MAX_FIXES = 3;
const BEAM_WIDTH = 5;
const MAX_SUSPECTS = 8;

// The six capture keys, in the order the guided scan visits them.
const FACE_KEYS = ['U', 'F', 'D', 'B', 'R', 'L'];

// rotate a row-major 4x4 grid k quarter-turns clockwise
export function rotGrid(grid, k) {
  let out = grid.slice();
  for (let n = 0; n < ((k % 4) + 4) % 4; n++) {
    const src = out;
    out = new Array(16);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[r * 4 + c] = src[(3 - c) * 4 + r];
  }
  return out;
}

function assemble(captures, rotations, rlSwapped, maps, swappedRLMaps, rotated) {
  const colors = new Array(96).fill(null);
  for (const key of FACE_KEYS) {
    const cap = captures[key];
    if (!cap) continue;
    const map = (rlSwapped && (key === 'R' || key === 'L')) ? swappedRLMaps[key] : maps[key];
    const grid = rotated[key][rotations[key]];
    for (let i = 0; i < 16; i++) colors[map[i]] = grid[i];
  }
  return colors;
}

// Enumerate rotation/swap combos ordered by how much they deviate from the
// captured reading, so the least surprising repair wins (and an already-valid
// state returns immediately at cost 0).
function allCombos() {
  const out = [];
  for (let n = 0; n < 4096; n++) {
    const rotations = {};
    let cost = 0;
    for (let i = 0; i < 6; i++) {
      const k = (n >> (i * 2)) & 3;
      rotations[FACE_KEYS[i]] = k;
      if (k) cost++;
    }
    out.push({ rotations, rlSwapped: false, cost });
    out.push({ rotations, rlSwapped: true, cost: cost + 1 });
  }
  out.sort((a, b) => a.cost - b.cost);
  return out;
}

// captures: { faceKey: {grid: 16 color letters, conf: 16 numbers} }
// -> null | { rotations, rlSwapped, fixes: [{idx, from, to}], colors }
export function repairState(captures, maps, swappedRLMaps) {
  if (!captures || FACE_KEYS.some((k) => !captures[k])) return null;

  const rotated = {};
  for (const key of FACE_KEYS) rotated[key] = [0, 1, 2, 3].map((k) => rotGrid(captures[key].grid, k));

  // ---- phase A: whole-face rotations x R/L swap ----
  let bestFallback = null;
  for (const combo of allCombos()) {
    const colors = assemble(captures, combo.rotations, combo.rlSwapped, maps, swappedRLMaps, rotated);
    const st = stateFromColors(colors);
    if (!st) continue;
    const v = validateState(st);
    if (v.ok) return { rotations: combo.rotations, rlSwapped: combo.rlSwapped, fixes: [], colors };
    if (!bestFallback || v.problems.length < bestFallback.problems.length) {
      bestFallback = { combo, colors, problems: v.problems };
    }
  }
  if (!bestFallback) return null;

  // ---- phase B: bounded sticker substitutions ----
  const conf = new Array(96).fill(1);
  const altByIdx = new Array(96).fill(null);
  for (const key of FACE_KEYS) {
    const map = (bestFallback.combo.rlSwapped && (key === 'R' || key === 'L')) ? swappedRLMaps[key] : maps[key];
    const k = bestFallback.combo.rotations[key];
    const cs = rotGrid(captures[key].conf, k);
    const as = captures[key].alt ? rotGrid(captures[key].alt, k) : null;
    for (let i = 0; i < 16; i++) {
      conf[map[i]] = cs[i];
      if (as) altByIdx[map[i]] = as[i];
    }
  }

  let beam = [{ colors: bestFallback.colors, problems: bestFallback.problems, fixes: [] }];
  for (let depth = 0; depth < MAX_FIXES; depth++) {
    const next = [];
    for (const node of beam) {
      for (const idx of suspectStickers(node.problems, node.colors, conf)) {
        for (const to of candidateColors(node.colors, idx, altByIdx)) {
          if (to === node.colors[idx]) continue;
          const colors = node.colors.slice();
          colors[idx] = to;
          const st = stateFromColors(colors);
          if (!st) continue;
          const v = validateState(st);
          const fixes = [...node.fixes, { idx, from: node.colors[idx], to }];
          if (v.ok) {
            return { rotations: bestFallback.combo.rotations, rlSwapped: bestFallback.combo.rlSwapped, fixes, colors };
          }
          if (v.problems.length < node.problems.length) {
            next.push({ colors, problems: v.problems, fixes });
          }
        }
      }
    }
    if (!next.length) break;
    next.sort((a, b) => a.problems.length - b.problems.length);
    beam = next.slice(0, BEAM_WIDTH);
  }
  return null;
}

function suspectStickers(problems, colors, conf) {
  const out = [];
  const seen = new Set();
  const add = (i) => { if (!seen.has(i) && conf[i] < 0.8) { seen.add(i); out.push(i); } };
  for (const p of problems) {
    if (p.stickers) for (const s of p.stickers) add(s);
  }
  // over-counted colors: their least confident stickers are the likely intruders
  for (const p of problems) {
    if (p.type === 'count' && p.count > 16) {
      const col = COLOR_OF_FACE[p.face];
      for (let i = 0; i < 96; i++) if (colors[i] === col) add(i);
    }
  }
  out.sort((a, b) => conf[a] - conf[b]);
  return out.slice(0, MAX_SUSPECTS);
}

function candidateColors(colors, idx, alt) {
  const cur = colors[idx];
  const out = [];
  if (NEAR[cur]) out.push(NEAR[cur]);
  if (alt && alt[idx] && !out.includes(alt[idx])) out.push(alt[idx]);
  const counts = {};
  for (const c of colors) counts[c] = (counts[c] || 0) + 1;
  for (const c of COLOR_KEYS) {
    if (c !== cur && (counts[c] || 0) < 16 && !out.includes(c)) out.push(c);
  }
  return out;
}
