// Precomputed piece-level move tables derived from the facelet engine.
import { FACES } from '../cube/geometry.js';
import { PERMS } from '../cube/moves.js';
import { CENTER_IDX, WINGS, EDGE_SLOTS } from '../cube/pieces.js';

// ---- centers: 24-entry state (face*4 + k), value = color code -------------
const CENTER_FACELETS = []; // 24 facelet indices in order
for (const f of FACES) for (const idx of CENTER_IDX[f]) CENTER_FACELETS.push(idx);

const centerPosOf = new Map(); // facelet index -> 0..23
CENTER_FACELETS.forEach((fi, i) => centerPosOf.set(fi, i));

export const CENTER_PERMS = {}; // token -> Int8Array(24): new[perm[i]] = old[i]
for (const tok of Object.keys(PERMS)) {
  const p = PERMS[tok];
  const cp = new Int8Array(24);
  let identity = true;
  for (let i = 0; i < 24; i++) {
    const dest = centerPosOf.get(p[CENTER_FACELETS[i]]);
    cp[i] = dest;
    if (dest !== i) identity = false;
  }
  CENTER_PERMS[tok] = identity ? null : cp; // null = doesn't touch centers
}

export function centersFromState(state) {
  const out = new Uint8Array(24);
  for (let i = 0; i < 24; i++) out[i] = state[CENTER_FACELETS[i]];
  return out;
}

// ---- wings: 24-entry state, value = ordered color pair (a*6+b) ------------
// Wing position i corresponds to WINGS[i]; colors ordered by sorted sticker index.
export const WING_MOVES = {}; // token -> [{to, swap}] indexed by from-position, or null
{
  const wingBySticker = new Map();
  WINGS.forEach((w, wi) => { wingBySticker.set(w.stickers[0], wi); wingBySticker.set(w.stickers[1], wi); });
  for (const tok of Object.keys(PERMS)) {
    const p = PERMS[tok];
    const mv = new Array(24);
    let identity = true;
    for (let wi = 0; wi < 24; wi++) {
      const [s1, s2] = WINGS[wi].stickers;
      const d1 = p[s1], d2 = p[s2];
      const dest = wingBySticker.get(d1);
      if (dest === undefined || wingBySticker.get(d2) !== dest) throw new Error('wing table bug for ' + tok);
      const swap = WINGS[dest].stickers[0] !== d1;
      mv[wi] = { to: dest, swap };
      if (dest !== wi || swap) identity = false;
    }
    WING_MOVES[tok] = identity ? null : mv;
  }
}

export function wingsFromState(state) {
  const out = new Uint8Array(24);
  for (let wi = 0; wi < 24; wi++) {
    const [s1, s2] = WINGS[wi].stickers;
    out[wi] = state[s1] * 6 + state[s2];
  }
  return out;
}

// a wing's two colors, swapped (its "flipped" reading)
export const flipWing = (v) => (v % 6) * 6 + ((v / 6) | 0);

export function applyWingMove(dest, src, mv) {
  for (let i = 0; i < 24; i++) {
    const { to, swap } = mv[i];
    const v = src[i];
    dest[to] = swap ? flipWing(v) : v;
  }
}

export function applyCenterMove(dest, src, perm) {
  if (!perm) { dest.set(src); return; }
  for (let i = 0; i < 24; i++) dest[perm[i]] = src[i];
}

// Edge slot -> its two wing position indices
export const SLOT_WINGS = EDGE_SLOTS.map((e) => e.wings);

export function pairedCount(wings) {
  let n = 0;
  for (const [a, b] of SLOT_WINGS) if (wings[a] === wings[b]) n++;
  return n;
}

// The outer-turn move set shared by every wing search: outer turns move edge
// slots as units, so they can never pair or unpair anything by themselves.
export const OUTER_TOKENS = [];
for (const b of ['U', 'D', 'F', 'B', 'R', 'L']) OUTER_TOKENS.push(b, b + "'", b + '2');
