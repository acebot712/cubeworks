// Turning the six per-face captures into one cube.
//
// Two consumers need this and used to each roll their own copy: the live cube
// the Confirm screen shows, and the input to the structural repair search.
// They differ only in whether repair corrections and manual edits are folded
// in, so both are expressed here over one shared spreading step.
import { CAPTURE_STEPS, CAPTURE_MAPS, SWAPPED_RL_MAPS } from './orientations.js';
import { resolveScanColors } from './resolveScan.js';
import { rotGrid } from '../cube/repair.js';

// Below CONF_WEAK a sticker is worth a human glance, so it joins the Confirm
// queue. Structural repairs land just under that line deliberately: they are
// applied, but never silently.
export const CONF_WEAK = 0.62;
const CONF_REPAIRED = 0.55;

function mapFor(faceKey, rlSwapped) {
  return (rlSwapped && (faceKey === 'R' || faceKey === 'L')) ? SWAPPED_RL_MAPS[faceKey] : CAPTURE_MAPS[faceKey];
}

// Spread each captured 4x4 grid onto its 96 home-frame facelets.
// pick(capture, faceKey) -> the 16-entry grid to spread, or null to skip.
function spread(rawCaptures, rlSwapped, pick) {
  const out = new Array(96).fill(null);
  for (const step of CAPTURE_STEPS) {
    const cap = rawCaptures[step.key];
    if (!cap) continue;
    const grid = pick(cap, step.key);
    if (!grid) continue;
    const map = mapFor(step.key, rlSwapped);
    for (let i = 0; i < 16; i++) out[map[i]] = grid[i];
  }
  return out;
}

// The raw RGB samples in home-frame order, at the rotation they were captured
// at. null until every face has been scanned with colour samples.
function rgbSamples(rawCaptures) {
  const rgb = spread(rawCaptures, false, (cap) => cap.rgb || null);
  return rgb.every(Boolean) ? rgb : null;
}

// The live cube: captured grids, re-estimated jointly once complete, then
// repair corrections and manual edits layered on top.
// -> { colors, conf, palette }  (palette is null until the cube is complete)
export function assembleCube({ rawCaptures, manualColors = {}, repair = null }) {
  const swapped = !!(repair && repair.rlSwapped);
  // A repair may re-interpret a face as captured at a different rotation.
  // rawCaptures itself is never mutated, so Undo is always possible.
  const turn = (key) => (repair && repair.rotations ? repair.rotations[key] || 0 : 0);
  const rotated = (field) => (cap, key) => {
    const grid = cap[field];
    if (!grid) return null;
    const k = turn(key);
    return k ? rotGrid(grid, k) : grid;
  };

  const colors = spread(rawCaptures, swapped, rotated('grid'));
  const conf = spread(rawCaptures, swapped, rotated('conf')).map((c) => c || 0);
  const rgbAll = spread(rawCaptures, swapped, rotated('rgb'));

  // Once every sticker has a raw RGB sample, re-estimate all 96 jointly. This
  // is where the scanner stops caring what colour the cube "should" be: a
  // pastel, neon or purple-for-blue cube resolves exactly like a standard one.
  let palette = null;
  if (rgbAll.every(Boolean)) {
    const resolved = resolveScanColors(rgbAll);
    palette = resolved.palette;
    for (let i = 0; i < 96; i++) {
      colors[i] = resolved.colors[i];
      conf[i] = resolved.conf[i];
    }
  }

  // Structural corrections land below the review threshold so they surface in
  // the Confirm queue for a human glance rather than being applied silently.
  for (const fix of (repair && repair.fixes) || []) {
    colors[fix.idx] = fix.to;
    conf[fix.idx] = CONF_REPAIRED;
  }
  for (const [idx, col] of Object.entries(manualColors)) {
    colors[idx] = col;
    conf[idx] = 1;
  }

  return { colors, conf, palette };
}

// Per-face captures for the repair search, in the colour vocabulary clustering
// produced. Clustering must run BEFORE the geometry search: it is
// position-independent, and repairing raw per-frame reads fails because those
// colours are still cast-corrupted. Returns null if the cube is not yet fully
// sampled.
export function capturesForRepair({ rawCaptures, manualColors = {} }) {
  const rgbAll = rgbSamples(rawCaptures);
  if (!rgbAll) return null;
  const resolved = resolveScanColors(rgbAll);

  const captures = {};
  for (const step of CAPTURE_STEPS) {
    const map = CAPTURE_MAPS[step.key];
    // a sticker the user has already corrected is ground truth: feed it in
    // as-is at full confidence so the search works around it
    const edited = (f) => manualColors[f] !== undefined;
    captures[step.key] = {
      grid: map.map((f) => (edited(f) ? manualColors[f] : resolved.colors[f])),
      conf: map.map((f) => (edited(f) ? 1 : resolved.conf[f])),
      alt: map.map((f) => resolved.alt[f]),
    };
  }
  return captures;
}
