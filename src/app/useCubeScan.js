// The scanned cube as data: what each face captured, what the user corrected,
// what the structural repair concluded, and the validated cube that falls out
// of all three.
import { useCallback, useEffect, useMemo, useState } from 'react';
import { FACE_INDEX } from '../cube/geometry.js';
import { stateFromColors } from '../cube/colorState.js';
import { validateState } from '../cube/validate.js';
import { repairState } from '../cube/repair.js';
import { assembleCube, capturesForRepair } from '../scan/assemble.js';
import { CAPTURE_STEPS, CAPTURE_MAPS, SWAPPED_RL_MAPS } from '../scan/orientations.js';
import { COLOR_KEYS } from '../state/colors.js';

const FACE_COUNT = CAPTURE_STEPS.length;

export function useCubeScan() {
  const [rawCaptures, setRawCaptures] = useState({});
  const [manualColors, setManualColors] = useState({});
  const [repair, setRepair] = useState(null);

  const { colors, conf, palette } = useMemo(
    () => assembleCube({ rawCaptures, manualColors, repair }),
    [rawCaptures, manualColors, repair]
  );

  const capturedCount = Object.keys(rawCaptures).length;
  const allCaptured = capturedCount === FACE_COUNT;
  const complete = colors.every(Boolean);

  const validation = useMemo(() => {
    if (!complete) return { ok: false, incomplete: true, problems: [] };
    return { ...validateState(stateFromColors(colors)), incomplete: false };
  }, [colors, complete]);

  const counts = useMemo(() => {
    const c = Object.fromEntries(COLOR_KEYS.map((k) => [k, 0]));
    for (const col of colors) if (col) c[col]++;
    return c;
  }, [colors]);

  // Structural error correction: an invalid complete state is usually a whole
  // face read at the wrong rotation, an R/L yaw the other way, or a couple of
  // misclassified stickers: all recoverable from the cube's own redundancy.
  useEffect(() => {
    if (!complete || validation.ok || validation.incomplete || repair) return;
    const captures = capturesForRepair({ rawCaptures, manualColors });
    if (!captures) return;
    setRepair(repairState(captures, CAPTURE_MAPS, SWAPPED_RL_MAPS) || { failed: true });
  }, [complete, validation, rawCaptures, manualColors, repair]);

  // Any change to the reading invalidates the repair conclusion drawn from it.
  const recordCapture = useCallback((faceKey, capture) => {
    setRawCaptures((rc) => ({ ...rc, [faceKey]: capture }));
    setRepair(null);
  }, []);

  const setManual = useCallback((idx, color) => {
    setManualColors((m) => ({ ...m, [idx]: color }));
  }, []);

  // Drop one face's capture and only ITS manual edits: corrections elsewhere
  // stand.
  const clearFace = useCallback((faceKey) => {
    setRawCaptures((rc) => {
      const next = { ...rc };
      delete next[faceKey];
      return next;
    });
    const base = FACE_INDEX[faceKey] * 16;
    setManualColors((m) => Object.fromEntries(
      Object.entries(m).filter(([idx]) => Number(idx) < base || Number(idx) >= base + 16)
    ));
    setRepair(null);
  }, []);

  const loadCaptures = useCallback((captures) => {
    setRawCaptures(captures);
    setManualColors({});
    setRepair(null);
  }, []);

  const restore = useCallback((saved) => {
    setRawCaptures(saved.rawCaptures);
    setManualColors(saved.manualColors || {});
    setRepair(saved.repair || null);
  }, []);

  const reset = useCallback(() => {
    setRawCaptures({});
    setManualColors({});
    setRepair(null);
  }, []);

  return {
    rawCaptures, manualColors, repair,
    colors, conf, palette,
    capturedCount, allCaptured, complete, validation, counts,
    recordCapture, setManual, clearFace, loadCaptures, restore, reset,
    // A dismissal must stick: clearing to null would just let the repair effect
    // fire again on the still-invalid state.
    dismissRepair: useCallback(() => setRepair({ dismissed: true }), []),
    reapplyRepair: useCallback(() => setRepair(null), []),
  };
}
