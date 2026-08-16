// The one conversion from a 96-entry list of colour letters (what a scan
// produces) to the engine's face-code state (what everything downstream
// consumes). Returns null if any sticker is missing or not a known colour.
import { FACE_INDEX } from './geometry.js';
import { FACE_OF_COLOR } from '../state/colors.js';

export function stateFromColors(colors) {
  if (!colors) return null;
  const state = new Uint8Array(96);
  for (let i = 0; i < 96; i++) {
    const face = FACE_OF_COLOR[colors[i]];
    if (face === undefined) return null;
    state[i] = FACE_INDEX[face];
  }
  return state;
}
