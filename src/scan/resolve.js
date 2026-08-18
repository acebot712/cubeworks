// Naming the six colour classes without a palette.
//
// Clustering tells us which stickers share a colour, but not which class is
// "up". A first attempt named classes by the colour of each face's centre
// stickers, that is valid on a 3x3, where centres are fixed, and WRONG on a
// 4x4, where the 2x2 centre blocks are scrambled like any other piece.
//
// The constraint that does hold is the cube's own geometry. The capture maps
// already pin every sticker to a facelet position, so the only freedom left is
// which class is called U, R, F, D, L, B, and of the 720 possible labellings,
// essentially one produces pieces that can physically exist (corners made of
// three mutually-adjacent colours, wings of two, no opposite-colour pairs).
// So we let the cube pick its own labelling and never look at a hue table.
import { FACES, FACELET_POS } from '../cube/geometry.js';
import { solvedState } from '../cube/state.js';
import { CORNERS } from '../cube/pieces.js';
import { validateState } from '../cube/validate.js';
import { nameCentroids } from './cluster.js';

// Corner chirality. Sticker counts and piece identity alone cannot tell a cube
// from its mirror image, nor from the same cube relabelled by a whole-cube
// rotation, both pass validateState. Twisting a corner rotates its three
// stickers but can never reflect them, so the RIGHT-HANDED cyclic order of a
// corner's three colours is an invariant of the physical piece. Matching that
// against the solved cube pins the labelling uniquely.
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];

// order a corner's 3 stickers so their outward normals form a right-handed triple
function rightHanded(stickers) {
  const [a, b, c] = stickers;
  const n = (i) => FACELET_POS[i].n;
  return dot(n(a), cross(n(b), n(c))) > 0 ? [a, b, c] : [a, c, b];
}
const RH_CORNERS = CORNERS.map((c) => rightHanded(c.stickers));

// reference: for each corner piece (keyed by its sorted face set) the
// right-handed cyclic sequence it shows when solved
const REF_CYCLE = (() => {
  const solved = solvedState();
  const m = new Map();
  for (const rh of RH_CORNERS) {
    const faces = rh.map((i) => FACES[solved[i]]);
    m.set([...faces].sort().join(''), faces);
  }
  return m;
})();

function chiralityOk(state) {
  for (const rh of RH_CORNERS) {
    const faces = rh.map((i) => FACES[state[i]]);
    const ref = REF_CYCLE.get([...faces].sort().join(''));
    if (!ref) return false;
    // must be a cyclic rotation of the reference, never a reflection
    const ok = [0, 1, 2].some((r) => ref[r] === faces[0] && ref[(r + 1) % 3] === faces[1] && ref[(r + 2) % 3] === faces[2]);
    if (!ok) return false;
  }
  return true;
}

function permutations(n) {
  const out = [];
  const cur = [];
  const used = new Array(n).fill(false);
  (function rec() {
    if (cur.length === n) { out.push(cur.slice()); return; }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true; cur.push(i);
      rec();
      cur.pop(); used[i] = false;
    }
  })();
  return out;
}
const PERMS6 = permutations(6);

// assign: 96 cluster ids (home-frame facelet order). centroids: 6 x [r,g,b].
// -> { faceOfCluster: 6 face letters, exact: true if the cube validated }
export function labelClusters(assign, centroids) {
  // Preference order for ties and for the fallback: how a human would name
  // these colours. Never used as a constraint, only to pick the friendliest of
  // otherwise-equivalent answers.
  const named = nameCentroids(centroids);              // e.g. ['W','R','G',...]
  const CANON_FACE = { W: 'U', R: 'R', G: 'F', Y: 'D', O: 'L', B: 'B' };
  const preferred = named.map((k) => FACES.indexOf(CANON_FACE[k]));

  const st = new Uint8Array(96);
  let best = null;
  for (const perm of PERMS6) {
    for (let i = 0; i < 96; i++) st[i] = perm[assign[i]];
    // validity says "these pieces could exist"; chirality says "and this is the
    // cube in the orientation it was actually scanned in, not its mirror or a
    // rotated relabelling". Both are needed to pin one answer.
    if (!validateState(st).ok) continue;
    if (!chiralityOk(st)) continue;
    // agreement with the human-friendly naming, purely as a tie-break
    let agree = 0;
    for (let k = 0; k < 6; k++) if (perm[k] === preferred[k]) agree++;
    if (!best || agree > best.agree) best = { perm: perm.slice(), agree };
    if (agree === 6) break;
  }

  if (best) return { faceOfCluster: best.perm.map((f) => FACES[f]), exact: true };
  // Nothing validated, the scan has real errors. Fall back to the friendly
  // naming so the Confirm screen can still show the cube and let the user fix
  // it; validation will report exactly what is impossible.
  return { faceOfCluster: preferred.map((f) => FACES[f]), exact: false };
}
