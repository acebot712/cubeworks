// Reading one video frame: locate the face, rectify it, and sample its 16
// cells. Everything here is imaging work, it touches <canvas> but knows
// nothing about React, screens or capture steps.
//
// All coordinates stay in raw, UNMIRRORED video pixels, matching CAPTURE_MAPS
// and the detector. The preview is drawn unmirrored too, so every surface the
// user sees agrees with the cube in their hands, no mirror math anywhere.
import { classify } from './classify.js';
import { createTracker } from './detect/tracker.js';
import { detectFaceInRoi } from './detect/search.js';
import { acquireFace, serviceAvailable } from './detect/acquire.js';
import { DETECT_W } from './detect/frame.js';

const RECT_N = 64;      // rectified face is RECT_N x RECT_N, so 16px per cell
const AWB_N = 8;        // white-balance thumbnail
const MOTION_N = 16;    // frame-diff grid, one tap per 4px of the rectified face
// A hand-placed box only snaps to a detected face if the search is confident;
// otherwise the user's own placement is respected exactly as drawn.
const MANUAL_SNAP_SCORE = 0.45;
// Model-assisted acquisition. The request is fired and forgotten, it takes
// ~1s while read() runs at 10Hz, and its box is then used as a search region
// for a few seconds. Purely an accelerator: without the service nothing here
// runs, and the classical search behaves exactly as before.
const ROI_TTL_MS = 4000;
const ACQUIRE_COOLDOWN_MS = 2500;

function canvas2d(width, height) {
  const el = document.createElement('canvas');
  el.width = width; el.height = height;
  return { el, ctx: el.getContext('2d', { willReadFrequently: true }) };
}

// Gray-world white balance measured on the whole camera frame: indoor lighting
// casts (warm bulbs) otherwise drag every sticker toward orange. Gains are
// clamped so a genuinely one-coloured scene cannot invent a correction.
function grayWorldGains(ctx, video) {
  ctx.drawImage(video, 0, 0, AWB_N, AWB_N);
  const d = ctx.getImageData(0, 0, AWB_N, AWB_N).data;
  const n = AWB_N * AWB_N;
  let mr = 0, mg = 0, mb = 0;
  for (let i = 0; i < n; i++) { mr += d[i * 4]; mg += d[i * 4 + 1]; mb += d[i * 4 + 2]; }
  mr /= n; mg /= n; mb /= n;
  const lum = (mr + mg + mb) / 3;
  if (lum <= 8) return { gains: [1, 1, 1], brightness: lum / 255 };
  const clamp = (x) => Math.max(0.75, Math.min(1.33, x));
  let peak = 0;
  for (let i = 0; i < n; i++) peak += Math.max(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]) / 255;
  return { gains: [clamp(lum / mr), clamp(lum / mg), clamp(lum / mb)], brightness: peak / n };
}

// Average a 5x5 neighbourhood at the centre of each of the 16 cells.
function readCells(data, gains) {
  const cells = [], confs = [], rgbs = [];
  let sumV = 0;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const cx = col * 16 + 8, cy = row * 16 + 8;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -4; dy <= 4; dy += 2) for (let dx = -4; dx <= 4; dx += 2) {
        const i = ((cy + dy) * RECT_N + (cx + dx)) * 4;
        r += data[i]; g += data[i + 1]; b += data[i + 2]; n++;
      }
      r = (r / n) * gains[0]; g = (g / n) * gains[1]; b = (b / n) * gains[2];
      sumV += Math.max(r, g, b) / 255;
      const cl = classify(r, g, b);
      cells.push(cl.key); confs.push(cl.conf); rgbs.push([r, g, b]);
    }
  }
  return { cells, confs, rgbs, lighting: sumV / 16 };
}

function grayGrid(data) {
  const gray = new Float32Array(MOTION_N * MOTION_N);
  for (let i = 0; i < gray.length; i++) {
    const px = ((Math.floor(i / MOTION_N) * 4) * RECT_N + (i % MOTION_N) * 4) * 4;
    gray[i] = (data[px] + data[px + 1] + data[px + 2]) / 3;
  }
  return gray;
}

// Stateful because detection tracks across frames and motion is a frame diff.
export function createFrameReader() {
  let detect = null, rect = null, awb = null;
  let tracker = null;
  let prevGray = null;
  let modelRoi = null;        // {box, at} suggested by the acquisition service
  let acquiring = false;
  let lastAcquire = 0;

  const ensure = () => {
    if (!rect) rect = canvas2d(RECT_N, RECT_N);
    if (!awb) awb = canvas2d(AWB_N, AWB_N);
    if (!tracker) tracker = createTracker();
  };

  return {
    // Forget the tracked pose and the motion history: used when the user
    // switches to or from a hand-placed frame.
    reset() {
      if (tracker) tracker.reset();
      prevGray = null;
      modelRoi = null;
    },

    // A hand-placed frame is treated as a REGION OF INTEREST, not as the final
    // answer: searching inside the user's box is 10-17x cheaper than a full
    // scan and measurably more accurate than either the box alone or a
    // full-frame search, so a roughly-placed box still snaps to the real face.
    // -> null (frame unusable) | {found:false,...} | {found:true,...}
    read(video, manualQuad) {
      if (!video || !video.videoWidth) return null;
      const vw = video.videoWidth, vh = video.videoHeight;
      ensure();

      const dh = Math.max(24, Math.round((DETECT_W * vh) / vw));
      if (!detect || detect.el.height !== dh) detect = canvas2d(DETECT_W, dh);
      let frame = null;
      try {
        detect.ctx.drawImage(video, 0, 0, DETECT_W, dh);
        frame = { data: detect.ctx.getImageData(0, 0, DETECT_W, dh).data, width: DETECT_W, height: dh };
      } catch { return null; }

      const scale = vw / DETECT_W;
      let det = null, snapped = null;
      if (manualQuad) {
        // search inside the user's box, in detection-frame coordinates
        const h = manualQuad.size / 2 / scale;
        const cx = manualQuad.cx / scale, cy = manualQuad.cy / scale;
        const hit = detectFaceInRoi(frame, [cx - h, cy - h, cx + h, cy + h]);
        if (hit && hit.score >= MANUAL_SNAP_SCORE) {
          snapped = { cx: hit.cx * scale, cy: hit.cy * scale, size: hit.size * scale, theta: hit.theta };
        }
      } else {
        det = tracker.update(frame);

        // Not found by the ordinary search: consult the acquisition service if
        // one is running, and meanwhile search inside whatever box it last gave.
        if (!det.found) {
          const now = Date.now();
          if (modelRoi && now - modelRoi.at < ROI_TTL_MS) {
            // refine inside EVERY proposal and keep the best-scoring quad: // the models disagree about which region is the face, and the
            // cube-face scorer is the only thing qualified to settle it
            let best = null;
            for (const p of modelRoi.proposals) {
              const hit = detectFaceInRoi(frame, p.box);
              if (hit && (!best || hit.score > best.score)) best = hit;
            }
            if (best && tracker.seed(best)) {
              det = tracker.update(frame);
              modelRoi = null;
            }
          } else if (serviceAvailable() !== false && !acquiring && now - lastAcquire > ACQUIRE_COOLDOWN_MS) {
            acquiring = true;
            acquireFace(frame)
              .then((proposals) => {
                if (proposals && proposals.length) modelRoi = { proposals, at: Date.now() };
              })
              .finally(() => { acquiring = false; lastAcquire = Date.now(); });
          }
        }
      }

      const quad = snapped || manualQuad || (det && det.found
        ? { cx: det.quad.cx * scale, cy: det.quad.cy * scale, size: det.quad.size * scale, theta: det.quad.theta }
        : null);

      let wb;
      try {
        wb = grayWorldGains(awb.ctx, video);
      } catch { return null; }

      if (!quad) {
        // Nothing located. Still report scene brightness: darkness is usually
        // the REASON the cube cannot be found, so this is when it matters most.
        prevGray = null;
        return { found: false, quad: null, vw, vh, vel: 0, lighting: wb.brightness, detScore: det ? det.score : 0 };
      }

      let data;
      try {
        const { ctx } = rect;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, RECT_N, RECT_N);
        ctx.translate(RECT_N / 2, RECT_N / 2);
        ctx.rotate(-quad.theta);
        ctx.scale(RECT_N / quad.size, RECT_N / quad.size);
        ctx.translate(-quad.cx, -quad.cy);
        ctx.drawImage(video, 0, 0);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        data = ctx.getImageData(0, 0, RECT_N, RECT_N).data;
      } catch { return null; }

      const { cells, confs, rgbs, lighting } = readCells(data, wb.gains);

      const gray = grayGrid(data);
      let motion = 0;
      if (prevGray) {
        let d = 0;
        for (let i = 0; i < gray.length; i++) d += Math.abs(gray[i] - prevGray[i]);
        motion = Math.min(1, d / gray.length / 26);
      }
      prevGray = gray;

      return {
        found: true, manual: !!manualQuad, quad, vw, vh,
        cells, confs, rgbs, lighting, motion,
        vel: det ? det.vel * scale : 0,
        detScore: det ? det.score : 0,
      };
    },
  };
}
