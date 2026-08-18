// The live half of scanning: run the frame reader on a timer, decide when the
// face is held steady enough to lock, and publish what the overlay draws.
//
// Faces advance only when the user says so: motion-based turn detection
// misfired constantly on handheld wobble and was removed.
import { useCallback, useEffect, useRef, useState } from 'react';
import { createFrameReader } from '../scan/sampler.js';
import { voteFrames } from '../scan/vote.js';

const TICK_MS = 100;
const VOTE_FRAMES = 10;         // frames retained for the capture vote
const DWELL_GAIN = 13;          // dwell % added per steady tick (~8 ticks to lock)
const DWELL_DECAY = 6;          // dwell % lost per unsteady tick
const STEADY_CELLS = 13;        // of 16 cells must match the previous frame
const CELL_TOLERANCE = 54;      // summed |dRGB| still counted as the same cell
const MOTION_LIMIT = 0.14;      // frame-diff above this reads as movement
const SLEW_LIMIT = 0.05;        // quad drift per tick, as a fraction of its size
const SEARCH_CAP_MS = 20000;    // how long "still looking" is allowed to climb

// enabled: the Scan screen is up with a working camera: running detection on
//   Confirm or Solve burned ten searches a second on data nothing consumed.
// active: scanning is running and there is a face still to capture.
// onLock(capture): receives the voted 16-cell grid when a face is read.
export function useScanLoop({ videoRef, enabled, active, onLock }) {
  const [live, setLive] = useState(null);
  const [dwell, setDwell] = useState(0);
  const [searchMs, setSearchMs] = useState(0);
  const [manualQuad, setManualQuad] = useState(null);

  const reader = useRef(null);
  const votes = useRef([]);
  const prevCells = useRef(null);
  // the tick reads these through refs so it never closes over stale state
  const latest = useRef({ active, dwell, manualQuad, onLock });
  latest.current = { active, dwell, manualQuad, onLock };

  const readFrame = useCallback(() => {
    if (!reader.current) reader.current = createFrameReader();
    return reader.current.read(videoRef.current, latest.current.manualQuad);
  }, [videoRef]);

  // Forget everything accumulated about the current face.
  const restart = useCallback(() => {
    votes.current = [];
    prevCells.current = null;
    setDwell(0);
    setSearchMs(0);
  }, []);

  const lock = useCallback((frame) => {
    const frames = frame ? [...votes.current, frame] : votes.current;
    if (!frames.length) return;
    latest.current.onLock(voteFrames(frames));
    restart();
  }, [restart]);

  // Capture whatever is on screen right now (the hand-placed-frame path).
  const lockNow = useCallback(() => {
    const frame = readFrame();
    if (frame && frame.found) lock(frame);
  }, [readFrame, lock]);

  const placeManualFrame = useCallback(() => {
    const v = videoRef.current;
    const vw = (v && v.videoWidth) || 1280;
    const vh = (v && v.videoHeight) || 720;
    setManualQuad({ cx: vw / 2, cy: vh / 2, size: 0.55 * Math.min(vw, vh), theta: 0 });
    setDwell(0);
  }, [videoRef]);

  const clearManualFrame = useCallback(() => {
    setManualQuad(null);
    setSearchMs(0);
    if (reader.current) reader.current.reset();
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    const id = setInterval(() => {
      const frame = readFrame();
      if (!frame) return;
      const s = latest.current;

      if (!frame.found) {
        setLive({ found: false, quad: null, cells: null, confs: null, rgbs: null, lighting: frame.lighting, moving: false, lock: 0, vw: frame.vw, vh: frame.vh });
        if (s.active) {
          setSearchMs((t) => Math.min(SEARCH_CAP_MS, t + TICK_MS));
          setDwell((d) => Math.max(0, d - DWELL_DECAY));
        }
        votes.current = [];
        prevCells.current = null;
        return;
      }
      setSearchMs(0);

      // The sampled window follows the cube, so frame-diff motion stays small
      // even while the cube moves: quad slew is the honest "it's moving"
      // signal, and the UI must report what the lock gate actually uses.
      const moving = frame.motion > MOTION_LIMIT || frame.vel > SLEW_LIMIT * frame.quad.size;
      setLive({
        found: true, manual: frame.manual, quad: frame.quad, vw: frame.vw, vh: frame.vh,
        cells: frame.cells, confs: frame.confs, rgbs: frame.rgbs,
        lighting: frame.lighting, moving, lock: frame.detScore,
      });

      // A hand-placed frame never auto-locks; the user presses capture.
      if (!s.active || frame.manual) { setDwell(0); return; }

      // Stability is measured on RAW RGB rather than on classified colour
      // names: naming depends on a palette, stability does not, so a cube with
      // unusual colours can still lock.
      let agree = 0;
      if (prevCells.current) {
        for (let i = 0; i < 16; i++) {
          const a = prevCells.current[i], b = frame.rgbs[i];
          if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]) < CELL_TOLERANCE) agree++;
        }
      }
      prevCells.current = frame.rgbs;

      if (moving || agree < STEADY_CELLS) {
        setDwell((d) => Math.max(0, d - DWELL_DECAY));
        if (moving) votes.current = [];
        return;
      }

      votes.current.push(frame);
      if (votes.current.length > VOTE_FRAMES) votes.current.shift();
      // Updaters must stay pure: scheduling the lock from inside one fired it
      // twice under StrictMode and double-advanced the face.
      setDwell((d) => Math.min(100, d + DWELL_GAIN));
      if (s.dwell + DWELL_GAIN >= 100) lock(null);
    }, TICK_MS);
    return () => clearInterval(id);
  }, [enabled, readFrame, lock]);

  return { live, dwell, searchMs, manualQuad, setManualQuad, restart, lockNow, placeManualFrame, clearManualFrame };
}
