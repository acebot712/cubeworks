// Frame-to-frame tracking: hysteresis, EMA smoothing and a rescan policy on top
// of the one-shot search.
//
// Acquiring needs a high score held over a STABLE pose, not just repeated high
// scores, that is what stops a lucky false positive from taking the lock.
// Losing is far more forgiving, so a brief occlusion does not drop it.
import { DEG } from './frame.js';
import { detectFace } from './search.js';

export const ACQUIRE_SCORE = 0.50;
const LOSE_SCORE = 0.30;
const ACQUIRE_TICKS = 3;
const LOSE_TICKS = 4;
const SMOOTH = 0.35;          // EMA on centre and size
const SMOOTH_THETA = 0.30;
const POSE_DRIFT = 0.15;      // centre within 0.15*size
const POSE_SCALE = 0.18;      // size within +/-18%
const POSE_THETA = 8 * DEG;
const RESCAN_EVERY = 20;      // ticks between full rescans of a degraded track
const RESCAN_MARGIN = 0.06;   // a rescan must beat the track by this to replace it

const samePose = (a, b) =>
  Math.hypot(a.cx - b.cx, a.cy - b.cy) <= POSE_DRIFT * b.size
  && Math.abs(a.size - b.size) <= POSE_SCALE * b.size
  && Math.abs(a.theta - b.theta) <= POSE_THETA;

export function createTracker() {
  let found = false;
  let quad = null;       // smoothed
  let anchor = null;     // pose of the first tick of the current acquiring run
  let goodRun = 0, badRun = 0, tick = 0;

  // While tracking, descend from the last pose. Periodically rescan from
  // scratch if the track has degraded, that guards against a false positive
  // holding the lock while the real face sits elsewhere in the frame.
  const locate = (frame) => {
    if (!found || !quad) return detectFace(frame, null);
    const tracked = detectFace(frame, quad);
    if (!tracked || tracked.score >= 0.9 * ACQUIRE_SCORE || tick % RESCAN_EVERY !== 0) return tracked;
    const full = detectFace(frame, null);
    return full && full.score > tracked.score + RESCAN_MARGIN ? full : tracked;
  };

  const tryAcquire = (raw) => {
    if (!raw || raw.score < ACQUIRE_SCORE) { goodRun = 0; anchor = null; return; }
    // re-anchor rather than zeroing: a cube moving into frame produces drifting
    // high-score hits that would otherwise never acquire
    if (anchor && samePose(raw, anchor)) goodRun++;
    else { goodRun = 1; anchor = { ...raw }; }
    if (goodRun < ACQUIRE_TICKS) return;
    found = true;
    quad = { ...raw }; // snap, no lerp on acquisition
    badRun = 0;
    anchor = null;
  };

  // -> the centre's movement this tick, for the "is it moving?" signal
  const trackOrDrop = (raw) => {
    if (!raw || raw.score < LOSE_SCORE) {
      badRun++;
      if (badRun >= LOSE_TICKS) { found = false; quad = null; goodRun = 0; }
      return 0; // during grace ticks keep the last quad (brief occlusion)
    }
    badRun = 0;
    const px = quad.cx, py = quad.cy;
    quad = {
      cx: quad.cx + (raw.cx - quad.cx) * SMOOTH,
      cy: quad.cy + (raw.cy - quad.cy) * SMOOTH,
      size: quad.size + (raw.size - quad.size) * SMOOTH,
      theta: quad.theta + (raw.theta - quad.theta) * SMOOTH_THETA,
    };
    return Math.hypot(quad.cx - px, quad.cy - py);
  };

  return {
    reset() { found = false; quad = null; anchor = null; goodRun = 0; badRun = 0; tick = 0; },

    // Hand the tracker a pose found by something else, a model's box refined
    // into a quad, or a hand-placed frame that snapped. It still has to earn
    // the lock through the normal acquire run, so a bad external suggestion
    // cannot pin a false positive: it only decides where to look first.
    seed(pose) {
      if (!pose || pose.score < ACQUIRE_SCORE) return false;
      anchor = { ...pose };
      goodRun = Math.max(goodRun, ACQUIRE_TICKS - 1);
      return true;
    },
    update(frame) {
      tick++;
      const raw = locate(frame);
      let vel = 0;
      if (found) vel = trackOrDrop(raw);
      else tryAcquire(raw);
      return { found, quad: found ? quad : null, raw, score: raw ? raw.score : 0, vel };
    },
  };
}
