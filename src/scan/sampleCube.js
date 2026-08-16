// A stand-in scan for when there is no camera: a real scrambled cube written
// through the capture pipeline as raw grids, so re-scanning, clustering and
// repair all behave exactly as they would on a real read.
import { FACES } from '../cube/geometry.js';
import { solvedState, applyMoves } from '../cube/state.js';
import { randomScramble } from '../cube/scramble.js';
import { COLOR_OF_FACE, COLORS } from '../state/colors.js';
import { CAPTURE_STEPS, CAPTURE_MAPS } from './orientations.js';

const SCRAMBLE_LEN = 40;

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// Sprinkle a deterministic minority of low-confidence reads so the Confirm
// queue has something to show.
const fakeConfidence = (facelet, cell) =>
  ((facelet * 7 + cell * 13) % 11 < 2 ? 0.42 + (cell % 3) * 0.05 : 0.86 + (cell % 5) * 0.02);

export function sampleCaptures() {
  const state = applyMoves(solvedState(), randomScramble(SCRAMBLE_LEN));
  const captures = {};
  for (const step of CAPTURE_STEPS) {
    const map = CAPTURE_MAPS[step.key];
    const grid = map.map((f) => COLOR_OF_FACE[FACES[state[f]]]);
    captures[step.key] = {
      grid,
      conf: map.map(fakeConfidence),
      rgb: grid.map((color) => hexToRgb(COLORS[color])),
    };
  }
  return captures;
}
