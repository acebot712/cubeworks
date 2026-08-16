// Piece-level validation of a scanned 4x4 state (before solving).
// Checks sticker counts, corner pieces (identity + twist sum), wing pieces
// (all 24 distinct valid wings) and center counts. Problems are reported as
// data; turning them into prose is the UI's job (see ui/problemText.js).
import { FACES } from './geometry.js';
import { CORNERS, WINGS, CENTER_IDX } from './pieces.js';

const OPPOSITE = { U: 'D', D: 'U', F: 'B', B: 'F', R: 'L', L: 'R' };

// the 8 valid corner color-sets of the standard scheme
const VALID_CORNERS = [];
for (const a of ['U', 'D']) for (const b of ['F', 'B']) for (const c of ['R', 'L']) {
  VALID_CORNERS.push([a, b, c].sort().join(''));
}

export function validateState(state) {
  const problems = [];

  // sticker counts
  const counts = {};
  for (const f of FACES) counts[f] = 0;
  for (let i = 0; i < 96; i++) counts[FACES[state[i]]]++;
  for (const f of FACES) {
    if (counts[f] !== 16) problems.push({ type: 'count', face: f, count: counts[f] });
  }

  // center counts (4 of each color among the 24 center facelets)
  const centerCounts = {};
  for (const f of FACES) centerCounts[f] = 0;
  for (const f of FACES) for (const idx of CENTER_IDX[f]) centerCounts[FACES[state[idx]]]++;
  for (const f of FACES) {
    if (centerCounts[f] !== 4) problems.push({ type: 'centers', face: f, count: centerCounts[f] });
  }

  // corners: valid piece identity per cubie + twist sum
  const seenCorners = new Map();
  let cornerBad = false;
  for (const corner of CORNERS) {
    const cols = corner.stickers.map((s) => FACES[state[s]]);
    const key = cols.slice().sort().join('');
    if (!VALID_CORNERS.includes(key) || new Set(cols).size !== 3) {
      problems.push({ type: 'corner', stickers: corner.stickers, colors: cols });
      cornerBad = true;
      continue;
    }
    seenCorners.set(key, (seenCorners.get(key) || 0) + 1);
  }
  if (!cornerBad) {
    for (const [key, n] of seenCorners) {
      if (n !== 1) problems.push({ type: 'cornerDup', colors: key, count: n });
    }
  }

  // wings: each of the 24 must be a valid (non-opposite, non-equal) pair and
  // each color pair must appear exactly twice (the two mirror wings)
  const wingCounts = new Map();
  for (const wing of WINGS) {
    const a = FACES[state[wing.stickers[0]]];
    const b = FACES[state[wing.stickers[1]]];
    if (a === b || OPPOSITE[a] === b) {
      problems.push({ type: 'wing', stickers: wing.stickers, colors: [a, b] });
      continue;
    }
    const key = [a, b].sort().join('');
    wingCounts.set(key, (wingCounts.get(key) || 0) + 1);
  }
  for (const [key, n] of wingCounts) {
    if (n !== 2) problems.push({ type: 'wingCount', colors: key, count: n });
  }

  return { ok: problems.length === 0, problems };
}
