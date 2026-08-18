// Finding the best window: coarse seeds, then hill-climbing.
//
// Seam alignment is a sharp peak in (cx, cy, size, theta), the basin is only a
// few px wide, so a fixed candidate lattice either misses it or costs a
// fortune. Coordinate descent with shrinking steps converges to sub-pixel
// precision cheaply, provided it starts somewhere sensible (see seeds.js).
import { DEG, quadFromPose, poseFromQuad, cropFrame } from './frame.js';
import { scoreWindow, scoreQuad } from './score.js';
import { coarseSeeds } from './seeds.js';

const MIN_SIZE = 12;      // px; below this a "face" is noise
const MIN_STEP = 0.4;     // px; stop refining position below this
const MAX_SWEEPS = 12;    // descent passes at one step size
const CORNER_STEP_FRAC = 0.06; // first corner nudge, as a fraction of face size
// Steps always shrink; a pass that found nothing shrinks them harder.
const DECAY_AFTER_PROGRESS = 0.6;
const DECAY_AFTER_STALL = 0.5;

// steps: { position, size, theta } in px / px / radians
function refine(frame, seed, steps) {
  let cur = { ...seed };
  if (cur.score === undefined) cur.score = scoreWindow(frame, cur.cx, cur.cy, cur.size, cur.theta);
  let stepP = steps.position, stepS = steps.size, stepT = steps.theta;

  const tryMove = (cx, cy, size, theta) => {
    if (size < MIN_SIZE) return;
    const s = scoreWindow(frame, cx, cy, size, theta);
    if (s > cur.score) cur = { cx, cy, size, theta, score: s };
  };

  while (stepP >= MIN_STEP) {
    let moved = false;
    for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
      const before = cur.score;
      tryMove(cur.cx - stepP, cur.cy, cur.size, cur.theta);
      tryMove(cur.cx + stepP, cur.cy, cur.size, cur.theta);
      tryMove(cur.cx, cur.cy - stepP, cur.size, cur.theta);
      tryMove(cur.cx, cur.cy + stepP, cur.size, cur.theta);
      tryMove(cur.cx, cur.cy, cur.size - stepS, cur.theta);
      tryMove(cur.cx, cur.cy, cur.size + stepS, cur.theta);
      tryMove(cur.cx, cur.cy, cur.size, cur.theta - stepT);
      tryMove(cur.cx, cur.cy, cur.size, cur.theta + stepT);
      if (cur.score <= before) break;
      moved = true;
    }
    const decay = moved ? DECAY_AFTER_PROGRESS : DECAY_AFTER_STALL;
    stepP *= decay; stepS *= decay; stepT *= decay;
  }
  return cur;
}

// Free the four corners. A cube held at an angle projects to a trapezoid, so
// the last few points of accuracy, and most of the accuracy on a 3/4 view: // live outside the rotated-square family the pose descent can reach.
function refineCorners(frame, pose) {
  let q = quadFromPose(pose.cx, pose.cy, pose.size, pose.theta);
  let score = scoreQuad(frame, q);
  if (score < pose.score) { score = pose.score; }
  let step = Math.max(MIN_STEP, pose.size * CORNER_STEP_FRAC);

  while (step >= MIN_STEP) {
    let moved = false;
    for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
      const before = score;
      for (let c = 0; c < 4; c++) {
        for (const [dx, dy] of [[-step, 0], [step, 0], [0, -step], [0, step]]) {
          const trial = q.map((p, i) => (i === c ? [p[0] + dx, p[1] + dy] : p));
          const s = scoreQuad(frame, trial);
          if (s > score) { score = s; q = trial; }
        }
      }
      if (score <= before) break;
      moved = true;
    }
    step *= moved ? DECAY_AFTER_PROGRESS : DECAY_AFTER_STALL;
  }
  return { ...poseFromQuad(q), corners: q, score };
}

// Search only inside a region someone else nominated (a model's box, a
// user-drawn frame). The comb runs on the crop; the descent then runs on the
// FULL frame, so the quad is free to settle slightly outside the box, a box
// around a tilted cube rarely bounds the face exactly.
// roi = [x0, y0, x1, y1] in frame pixels. -> same shape as detectFace, or null
export function detectFaceInRoi(frame, roi, pad = 0.12) {
  const [x0, y0, x1, y1] = roi;
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  const crop = cropFrame(frame, x0 - px, y0 - py, x1 + px, y1 + py);
  if (!crop) return null;

  let best = null;
  for (const seed of coarseSeeds(crop)) {
    const moved = { ...seed, cx: seed.cx + crop.offsetX, cy: seed.cy + crop.offsetY };
    const r = refine(frame, { ...moved, score: undefined }, {
      position: Math.max(2, moved.size / 12), size: moved.size * 0.06, theta: 3 * DEG,
    });
    if (!best || r.score > best.score) best = r;
  }
  // No comb seed inside the box: fall back to descending from the box itself,
  // which is what happens when the face is too small for the crop's comb range.
  if (!best) {
    const size = Math.min(x1 - x0, y1 - y0) * 0.86;
    best = refine(frame, { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, size, theta: 0, score: undefined }, {
      position: Math.max(2, size / 10), size: size * 0.10, theta: 4 * DEG,
    });
  }
  return best ? refineCorners(frame, best) : null;
}

// One-shot search. hint = last known {cx, cy, size, theta}, which turns this
// into a cheap short descent from the previous pose instead of a full scan.
// -> {cx, cy, size, theta, corners, score} or null
export function detectFace(frame, hint = null) {
  if (hint) {
    const pose = refine(frame, { ...hint, score: undefined }, {
      position: Math.max(1.5, hint.size / 16), size: hint.size * 0.05, theta: 3 * DEG,
    });
    return refineCorners(frame, pose);
  }

  let best = null;
  for (const seed of coarseSeeds(frame)) {
    const r = refine(frame, seed, {
      position: Math.max(2, seed.size / 12), size: seed.size * 0.06, theta: 3 * DEG,
    });
    if (!best || r.score > best.score) best = r;
  }
  return best ? refineCorners(frame, best) : null;
}
