// What the Scan screen tells the user, as data.
//
// Two UX invariants live here and are easiest to keep honest when the whole
// decision is one pure function: there is exactly ONE primary action on screen
// at a time, and the hint bar is informational only.
import { ACCENT, ACCENT_INK, INK, WARN, INFO } from '../theme.js';

const STUCK_MS = 4000;   // after this long with no face, offer the manual frame

const TONE = {
  accent: { chipBg: 'rgba(79,227,193,0.14)', chipFg: ACCENT, fg: INK, border: 'rgba(255,255,255,0.09)' },
  info: { chipBg: 'rgba(44,107,232,0.18)', chipFg: INFO, fg: INK, border: 'rgba(255,255,255,0.09)' },
  warn: { chipBg: 'rgba(255,158,82,0.16)', chipFg: WARN, fg: INK, border: 'rgba(255,255,255,0.09)' },
  alarm: { chipBg: 'rgba(255,122,26,0.16)', chipFg: WARN, fg: WARN, border: 'rgba(255,122,26,0.3)' },
};

const ACTION_STYLE = {
  go: { bg: ACCENT, fg: ACCENT_INK, glow: 'rgba(79,227,193,0.4)' },
  turn: { bg: INFO, fg: '#0A1730', glow: 'rgba(127,168,245,0.4)' },
  manual: { bg: WARN, fg: '#2A1400', glow: 'rgba(255,158,82,0.35)' },
};

const action = (label, icon, kind, onClick) => ({ label, icon, onClick, ...ACTION_STYLE[kind] });
const hint = (icon, title, sub, tone = 'accent') => ({ icon, title, sub, ...TONE[tone] });

// state: { allCaptured, scanning, awaitingTurn, manualQuad, found, lowLight,
//          moving, dwell, searchMs, step }
// handlers: { goReview, startScanning, continueScan, lockNow, placeManualFrame }
export function primaryAction(state, handlers) {
  const { allCaptured, scanning, awaitingTurn, manualQuad, found, searchMs } = state;
  if (allCaptured) return action('Confirm the read', '✓', 'go', handlers.goReview);
  if (!scanning) return action('Start scanning', '▶', 'go', handlers.startScanning);
  if (awaitingTurn) return action("I've turned it: read this face", '↻', 'turn', handlers.continueScan);
  if (manualQuad) return action('Capture this frame', '⧉', 'go', handlers.lockNow);
  if (!found && searchMs >= STUCK_MS) return action('Place the frame myself', '✋', 'manual', handlers.placeManualFrame);
  return null;
}

export function hintFor(state) {
  const { allCaptured, scanning, awaitingTurn, manualQuad, found, lowLight, moving, dwell, searchMs, step } = state;

  if (allCaptured) return hint('✓', 'All six faces captured', 'Move on to confirm the read');
  if (!scanning) return hint('▶', 'Hold the cube up, then start', "I'll find the face wherever it is in view");
  if (awaitingTurn) return hint('↻', 'Locked: turn the cube', step.instr, 'info');
  if (manualQuad) return hint('✋', 'Line the box up with one face', 'Drag to move, scroll to resize, then capture', 'warn');

  if (!found) {
    if (lowLight) {
      return hint('☾', 'Too dark to find the cube',
        'Add light or face a window: I cannot separate the colours this dark', 'alarm');
    }
    return hint('⌕', 'Looking for the cube…', searchMs >= STUCK_MS
      ? 'Still nothing: try a plainer background, or place the frame yourself'
      : 'Show one face flat to the camera, holding it by the edges');
  }

  if (lowLight) return hint('☾', 'Too dark to read colors', 'Add light, or move toward a window', 'alarm');
  if (moving) return hint('⇢', 'Hold still', 'I need a steady frame to lock this face', 'warn');
  if (dwell > 5) return hint('◉', 'Reading…', 'Hold it steady: got it');
  return hint('⬒', step.instr, step.sub);
}
