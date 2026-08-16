// Animated CSS-3D 4x4x4 cube. Shows `states[idx]`; when idx advances by one
// it animates `moves[idx-1]` as a real layer rotation, then commits.
import React, { useEffect, useRef, useState } from 'react';
import { FACELET_POS } from '../cube/geometry.js';
import { moveInfo } from '../cube/moveTable.js';

const FACE_ROT = {
  F: '', B: 'rotateY(180deg)', R: 'rotateY(90deg)', L: 'rotateY(-90deg)',
  U: 'rotateX(90deg)', D: 'rotateX(-90deg)',
};
// engine-positive quarter turns -> CSS rotation signs (CSS y is down)
const CSS_AXIS = { x: ['rotateX', 1], y: ['rotateY', -1], z: ['rotateZ', 1] };
const AXIS_IDX = { x: 0, y: 1, z: 2 };

const YAW = -32;
const PITCH = -22;
const DURATION = 280;
const GLOW = '#4FE3C1';

export default function Cube3D({ states, moves, idx, colorOf, labelOf, labels = false, size = 300 }) {
  const [shown, setShown] = useState(idx);
  const [anim, setAnim] = useState(null); // {token, angleOn}
  const timer = useRef(null);
  const shownRef = useRef(idx);
  shownRef.current = shown;

  useEffect(() => {
    if (idx === shownRef.current) return;
    clearTimeout(timer.current);
    if (idx === shownRef.current + 1 && moves && moves[shownRef.current]) {
      const token = moves[shownRef.current];
      setAnim({ token, angleOn: false });
      requestAnimationFrame(() => requestAnimationFrame(() => {
        setAnim((a) => (a ? { ...a, angleOn: true } : a));
      }));
      timer.current = setTimeout(() => {
        setAnim(null);
        setShown(idx);
      }, DURATION + 40);
    } else {
      setAnim(null);
      setShown(idx);
    }
  }, [idx, moves]);

  // clamp shown to valid range if states changed
  const state = states[Math.min(shown, states.length - 1)] || states[0];
  const s = size / 4;

  let animInfo = null;
  if (anim) {
    const info = moveInfo(anim.token);
    if (info) {
      const [fn, sign] = CSS_AXIS[info.axis];
      animInfo = {
        axisIdx: AXIS_IDX[info.axis],
        layers: new Set(info.layers),
        css: `${fn}(${anim.angleOn ? sign * info.turns * 90 : 0}deg)`,
      };
    }
  }
  // highlight the next move's layer when idle
  let nextInfo = null;
  if (!anim && moves && moves[shown]) {
    const info = moveInfo(moves[shown]);
    if (info) nextInfo = { axisIdx: AXIS_IDX[info.axis], layers: new Set(info.layers) };
  }

  const stickers = [];
  for (let i = 0; i < 96; i++) {
    const { face, r, c, p } = FACELET_POS[i];
    const cellX = (c - 1.5) * s;
    const cellY = (r - 1.5) * s;
    const inAnim = animInfo && animInfo.layers.has(p[animInfo.axisIdx]);
    const inNext = nextInfo && nextInfo.layers.has(p[nextInfo.axisIdx]);
    const pre = inAnim ? animInfo.css + ' ' : '';
    const color = colorOf(state[i]);
    stickers.push(
      <div
        key={i}
        style={{
          position: 'absolute', left: '50%', top: '50%',
          width: s - 5, height: s - 5, marginLeft: -(s - 5) / 2, marginTop: -(s - 5) / 2,
          transform: `${pre}${FACE_ROT[face]} translate3d(${cellX}px, ${cellY}px, ${size / 2 + 1}px)`,
          transition: inAnim && anim.angleOn ? `transform ${DURATION}ms cubic-bezier(.35,0,.25,1)` : 'none',
          background: color,
          borderRadius: 4,
          boxShadow: inAnim || inNext
            ? `0 0 0 2px ${GLOW}, 0 0 14px rgba(79,227,193,0.45)`
            : 'inset 0 0 0 1px rgba(0,0,0,0.28)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700,
          color: 'rgba(0,0,0,0.5)',
          backfaceVisibility: 'hidden',
        }}
      >
        {labels ? labelOf(state[i]) : ''}
      </div>
    );
  }

  return (
    <div style={{ perspective: 1400, width: size, height: size, position: 'relative' }}>
      <div
        style={{
          position: 'absolute', inset: 0, transformStyle: 'preserve-3d',
          transform: `rotateX(${PITCH}deg) rotateY(${YAW}deg)`,
          transition: 'transform 650ms cubic-bezier(.4,0,.2,1)',
        }}
      >
        {/* dark core so gaps between stickers read as the cube body */}
        <div style={{
          position: 'absolute', left: '50%', top: '50%', width: size - 6, height: size - 6,
          marginLeft: -(size - 6) / 2, marginTop: -(size - 6) / 2,
          transform: 'translateZ(0px)', background: 'transparent',
        }} />
        {['F', 'B', 'R', 'L', 'U', 'D'].map((f) => (
          <div key={f} style={{
            position: 'absolute', left: '50%', top: '50%', width: size, height: size,
            marginLeft: -size / 2, marginTop: -size / 2,
            transform: `${FACE_ROT[f]} translateZ(${size / 2 - 1}px)`,
            background: '#0B0C0E', borderRadius: 6,
          }} />
        ))}
        {stickers}
      </div>
    </div>
  );
}
