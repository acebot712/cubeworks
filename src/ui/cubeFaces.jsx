// The two cube widgets every screen reuses: a face's 4x4 sticker grid, and the
// six-swatch colour picker. Four hand-rolled copies of the grid used to live in
// the screens, each with slightly different highlight rules.
import React from 'react';
import { FACES, FACE_INDEX } from '../cube/geometry.js';
import { COLORS, COLOR_KEYS, NAMES } from '../state/colors.js';
import { Btn } from './primitives.jsx';
import { ACCENT, BG_EMPTY, mono } from './theme.js';

// Face letters in the order the guided scan and the net present them.
export const FACE_LIST = [
  { key: 'U', name: 'Up face' }, { key: 'F', name: 'Front face' },
  { key: 'D', name: 'Down face' }, { key: 'B', name: 'Back face' },
  { key: 'R', name: 'Right face' }, { key: 'L', name: 'Left face' },
];
export const FACE_NAME = { U: 'Up', F: 'Front', D: 'Down', B: 'Back', R: 'Right', L: 'Left' };

export const faceOfFacelet = (idx) => FACES[(idx / 16) | 0];
export const faceBase = (faceKey) => FACE_INDEX[faceKey] * 16;

// A cube's own measured palette when it has one, the defaults until then.
export const colorLookup = (palette) => (key) => (palette && palette[key]) || COLORS[key];

// One face of the cube.
//   highlight(idx) -> falsy | 'selected' | 'weak' | 'flagged'
export function StickerGrid({
  faceKey, colors, colorOf, labels = false, highlight, onPick, cell = null, gap = 3, radius = 4,
}) {
  const base = faceBase(faceKey);
  const RING = {
    selected: `0 0 0 3px ${ACCENT}`,
    focus: `0 0 0 3px ${ACCENT}, 0 0 20px rgba(79,227,193,.5)`,
    weak: '0 0 0 2px rgba(255,122,26,0.7)',
  };
  return (
    <div style={{
      display: 'grid', gap, height: '100%',
      gridTemplateColumns: `repeat(4, ${cell ? `${cell}px` : '1fr'})`,
      gridTemplateRows: `repeat(4, ${cell ? `${cell}px` : '1fr'})`,
    }}>
      {Array.from({ length: 16 }, (_, i) => {
        const idx = base + i;
        const col = colors[idx];
        const mark = highlight ? highlight(idx) : null;
        return (
          <div
            key={i}
            onClick={onPick ? () => onPick(idx) : undefined}
            style={{
              borderRadius: radius,
              background: col ? colorOf(col) : BG_EMPTY,
              cursor: onPick ? 'pointer' : 'default',
              boxShadow: RING[mark] || 'inset 0 0 0 1px rgba(0,0,0,0.28)',
              animation: mark === 'flagged' ? 'cw-ring 1.4s ease-in-out infinite' : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              ...mono, fontSize: 11, fontWeight: 700, color: 'rgba(0,0,0,0.5)',
            }}
          >
            {labels && col ? col : ''}
          </div>
        );
      })}
    </div>
  );
}

// The six swatches. `current` is the colour of whatever is selected, so the
// active swatch can show a ring; a null selection disables the whole row.
export function ColorPicker({ colorOf, current, onPick, disabled, size = 30, stretch = false }) {
  return (
    <>
      {COLOR_KEYS.map((key) => (
        <Btn
          key={key}
          disabled={disabled}
          label={NAMES[key]}
          onClick={() => onPick(key)}
          style={{
            ...(stretch ? { flex: 1 } : { width: size }),
            height: size, borderRadius: 7, background: colorOf(key),
            opacity: disabled ? 0.3 : 1,
            border: `2px solid ${!disabled && current === key ? '#FFFFFF' : 'rgba(255,255,255,0.14)'}`,
          }}
        />
      ))}
    </>
  );
}
