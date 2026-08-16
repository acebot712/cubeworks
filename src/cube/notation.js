// Move-sequence algebra: parse, invert, simplify, and translate to the notation
// a human reads. Operates on tokens alone — no cube state is ever touched.
import { MOVE_DEFS, tokenBase, tokenTurns, tokenFromTurns, sameAxis, layersOverlap } from './moveTable.js';

export function parseMoves(s) {
  if (Array.isArray(s)) return s;
  return s.trim().split(/\s+/).filter(Boolean);
}

export function invertMoves(moves) {
  return parseMoves(moves).slice().reverse().map((m) => {
    if (m.endsWith('2')) return m;
    if (m.endsWith("'")) return m.slice(0, -1);
    return m + "'";
  });
}

// Merge consecutive moves of the same base and cancel identities. Moves on the
// same axis with disjoint layers commute, so the merge hops over them: that is
// what turns "R L R'" into "L".
export function simplifyMoves(moves) {
  let list = parseMoves(moves).slice();
  let changed = true;
  while (changed) {
    changed = false;
    const out = [];
    for (const m of list) {
      const base = tokenBase(m);
      let j = out.length - 1;
      let merged = false;
      while (j >= 0) {
        const prev = tokenBase(out[j]);
        if (prev === base) {
          const turns = (tokenTurns(out[j], prev) + tokenTurns(m, base)) % 4;
          const tok = tokenFromTurns(base, turns);
          out.splice(j, 1);
          if (tok) out.splice(j, 0, tok);
          merged = true;
          changed = true;
          break;
        }
        if (!sameAxis(prev, base) || layersOverlap(prev, base)) break;
        j--; // commutes: keep looking further back
      }
      if (!merged) out.push(m);
    }
    list = out;
  }
  return list;
}

// Internal tokens to user-facing WCA-ish ones: a bare inner slice becomes a
// wide+outer pair (r == Rw R'). Rotations pass through unchanged.
// the bare inner slices u/d/r/l/f/b — single lower-case bases that are not
// whole-cube rotations
const INNER_SLICES = new Set(Object.keys(MOVE_DEFS).filter(
  (b) => b.length === 1 && b === b.toLowerCase() && MOVE_DEFS[b][1].length !== 4
));

export function toUserMoves(moves) {
  const out = [];
  for (const m of parseMoves(moves)) {
    const base = tokenBase(m);
    if (!INNER_SLICES.has(base)) { out.push(m); continue; }
    const wide = base.toUpperCase();
    const suffix = m.slice(base.length);
    // r == Rw R'  ->  r' == Rw' R,  r2 == Rw2 R2
    if (suffix === '2') out.push(wide + 'w2', wide + '2');
    else if (suffix === "'") out.push(wide + "w'", wide);
    else out.push(wide + 'w', wide + "'");
  }
  return simplifyMoves(out);
}
