// Turning a move token into something a person can follow. Presentation only: // the solver never sees these strings.

const LAYER_NAME = {
  U: 'Top layer', Uw: 'Top two layers', D: 'Bottom layer', Dw: 'Bottom two layers',
  R: 'Right column', Rw: 'Right two columns', L: 'Left column', Lw: 'Left two columns',
  F: 'Front face', Fw: 'Front two layers', B: 'Back face', Bw: 'Back two layers',
  x: 'Whole cube', y: 'Whole cube', z: 'Whole cube',
};

// [clockwise-sense direction, its inverse]
const DIRECTION = {
  U: ['left', 'right'], Uw: ['left', 'right'], D: ['right', 'left'], Dw: ['right', 'left'],
  R: ['up', 'down'], Rw: ['up', 'down'], L: ['down', 'up'], Lw: ['down', 'up'],
  F: ['clockwise', 'counter-clockwise'], Fw: ['clockwise', 'counter-clockwise'],
  B: ['counter-clockwise from your view', 'clockwise from your view'],
  Bw: ['counter-clockwise from your view', 'clockwise from your view'],
  x: ['tilt back', 'tilt forward'], y: ['spin left', 'spin right'], z: ['roll right', 'roll left'],
};

// [clockwise glyph, counter-clockwise glyph]; half turns always show ⇄ or ⇅.
const GLYPHS = {
  U: ['↺', '↻', '⇄'], D: ['↻', '↺', '⇄'],
  R: ['↑', '↓', '⇅'], L: ['↓', '↑', '⇅'],
  F: ['↻', '↺', '⇄'], B: ['↺', '↻', '⇄'],
};

const baseOf = (token) => token.replace(/['2]/g, '');

export function describeMove(token) {
  const base = baseOf(token);
  const name = LAYER_NAME[base] || base;
  if (token.endsWith('2')) return `${name}, half turn`;
  const dirs = DIRECTION[base] || ['clockwise', 'counter-clockwise'];
  return `${name}, ${token.endsWith("'") ? dirs[1] : dirs[0]}`;
}

export function arrowFor(token) {
  if (!token) return '↻';
  // wide turns rotate the same way as their outer face
  const glyphs = GLYPHS[baseOf(token).replace('w', '')];
  if (!glyphs) return '↻';
  return token.endsWith('2') ? glyphs[2] : token.endsWith("'") ? glyphs[1] : glyphs[0];
}
