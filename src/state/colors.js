// Colour vocabulary shared by scan, review and solve.
// Internal solver states use face letters (URFDLB); scanning uses colour
// letters (WYGBRO). Standard western scheme: U=white F=green R=red.
//
// COLOR_OF_FACE is the single source of truth; the reverse map and the key
// list are derived from it so the directions can never drift apart.
export const COLOR_OF_FACE = { U: 'W', R: 'R', F: 'G', D: 'Y', L: 'O', B: 'B' };
export const FACE_OF_COLOR = Object.fromEntries(
  Object.entries(COLOR_OF_FACE).map(([face, color]) => [color, face])
);

// Default hexes and names, used for display until a scan measures the cube's
// own palette. Order is the swatch order the pickers show.
export const COLORS = { W: '#F2F3F5', Y: '#FFD028', G: '#23B15A', B: '#2C6BE8', R: '#E8402A', O: '#FF7A1A' };
export const NAMES = { W: 'White', Y: 'Yellow', G: 'Green', B: 'Blue', R: 'Red', O: 'Orange' };
export const COLOR_KEYS = Object.keys(COLORS);

// likely camera mix-ups
export const NEAR = { R: 'O', O: 'R', Y: 'W', W: 'Y', G: 'B', B: 'G' };

// The colour a face wears when solved, named for prose.
export const faceColorName = (face) => NAMES[COLOR_OF_FACE[face]].toLowerCase();
