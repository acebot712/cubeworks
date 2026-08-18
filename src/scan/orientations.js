// Capture-step orientation mapping.
//
// During scanning the camera-facing face occupies the engine's F slot.
// Each guided step has a whole-cube rotation sequence describing how the cube
// is being held at that moment; the map from on-screen grid cells to
// cube-frame facelet indices is derived with a label-tracking cube, so no
// hand-written per-face rotation tables exist anywhere.
import { FACE_INDEX } from '../cube/geometry.js';
import { PERMS } from '../cube/moves.js';
import { parseMoves } from '../cube/notation.js';

function labelsAfter(moves) {
  let labels = Int16Array.from({ length: 96 }, (_, i) => i);
  for (const tok of parseMoves(moves)) {
    const perm = PERMS[tok];
    const next = new Int16Array(96);
    for (let i = 0; i < 96; i++) next[perm[i]] = labels[i];
    labels = next;
  }
  return labels;
}

// screen grid (row-major 16 cells, unmirrored camera view) -> cube facelet idx
function captureMap(orientMoves) {
  const labels = labelsAfter(orientMoves);
  const fi = FACE_INDEX.F;
  const map = new Array(16);
  for (let i = 0; i < 16; i++) map[i] = labels[fi * 16 + i];
  return map;
}

export const CAPTURE_STEPS = [
  {
    key: 'U', orient: ["x'"],
    instr: 'Show the top face', sub: 'Tip the cube back so its top points at the camera',
  },
  {
    key: 'F', orient: [],
    instr: 'Roll the cube toward you', sub: 'The face that was facing you comes into view',
  },
  {
    key: 'D', orient: ['x'],
    instr: 'Roll toward you again', sub: 'Keep the same grip: third face in the loop',
  },
  {
    key: 'B', orient: ['x2'],
    instr: 'One more roll toward you', sub: 'Fourth face in this loop',
  },
  {
    key: 'R', orient: ['y'],
    instr: 'Back upright, then spin left', sub: 'Roll once more, then turn the right-hand face to the camera',
  },
  {
    key: 'L', orient: ["y'"],
    instr: 'Spin to the opposite side', sub: 'Half-way around, the last face',
  },
];

export const CAPTURE_MAPS = Object.fromEntries(
  CAPTURE_STEPS.map((s) => [s.key, captureMap(s.orient)])
);

// If the user yawed the other way on steps 5/6, R and L captures are swapped:
// the "R" grid actually shows L (held in y' orientation) and vice versa.
export const SWAPPED_RL_MAPS = {
  R: captureMap(["y'"]),
  L: captureMap(['y']),
};
