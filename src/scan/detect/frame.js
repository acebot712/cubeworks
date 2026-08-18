// Reading a downscaled video frame, and the window-local coordinate system the
// scorer works in.
//
// A frame is {data: Uint8ClampedArray RGBA, width, height} in raw, UNMIRRORED
// camera orientation, the same orientation CAPTURE_MAPS are defined against.
// A candidate window is a square of side `size` centred at (cx, cy) and rotated
// by `theta`; inside it, (u, v) run from -size/2 to +size/2.

export const DEG = Math.PI / 180;

// Width the video is downscaled to before detection. A face filling only ~11%
// of frame height (which is how people actually hold a cube) is 12px at 192,
// i.e. 3px per sticker with sub-pixel seams. There is no signal left to find at
// that size, so the search resolution has to carry it.
export const DETECT_W = 640;

export const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function tap(frame, x, y) {
  const xi = x < 0 ? 0 : x >= frame.width ? frame.width - 1 : x | 0;
  const yi = y < 0 ? 0 : y >= frame.height ? frame.height - 1 : y | 0;
  const i = (yi * frame.width + xi) * 4;
  return [frame.data[i], frame.data[i + 1], frame.data[i + 2]];
}

export function luma(c) { return (2 * c[0] + 3 * c[1] + c[2]) / 6; }

// Difference in chromaticity, so it survives a brightness change.
export function chromaDiff(a, b) {
  const sa = a[0] + a[1] + a[2] + 1e-6, sb = b[0] + b[1] + b[2] + 1e-6;
  return (
    Math.abs((a[0] / sa - b[0] / sb)) +
    Math.abs((a[1] / sa - b[1] / sb)) +
    Math.abs((a[2] / sa - b[2] / sb))
  ) * 255;
}

// -> { px(u, v) -> [r,g,b],  inFrame(u, v) -> boolean }
export function windowSampler(frame, cx, cy, size, theta) {
  const cos = Math.cos(theta), sin = Math.sin(theta);
  const toImage = (u, v) => [cx + u * cos - v * sin, cy + u * sin + v * cos];
  return {
    px: (u, v) => { const [x, y] = toImage(u, v); return tap(frame, x, y); },
    inFrame: (u, v) => {
      const [x, y] = toImage(u, v);
      return x >= 0 && y >= 0 && x < frame.width && y < frame.height;
    },
  };
}

// Centre of cell `i` (0..3) along one axis, in window coords.
export const cellCentre = (i, size) => ((i + 0.5) / 4 - 0.5) * size;
// Position of the i-th interior grid line (1..3), in window coords.
export const gridLine = (i, size) => (i / 4 - 0.5) * size;

// Box-downsample a frame. The coarse search only needs enough resolution to
// find the seam PERIOD; the fine descent needs full resolution to place the
// corners. Running both at full size costs 4x for nothing.
export function downsample(frame, targetW) {
  const { data, width: w, height: h } = frame;
  if (targetW >= w) return frame;
  const tw = targetW, th = Math.max(1, Math.round((h * targetW) / w));
  const out = new Uint8ClampedArray(tw * th * 4);
  const sx = w / tw, sy = h / th;
  for (let y = 0; y < th; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < tw; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1 && yy < h; yy++) {
        for (let xx = x0; xx < x1 && xx < w; xx++) {
          const j = (yy * w + xx) * 4;
          r += data[j]; g += data[j + 1]; b += data[j + 2]; n++;
        }
      }
      const o = (y * tw + x) * 4;
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return { data: out, width: tw, height: th };
}

// Crop to a region of interest. When something else has already said roughly
// where the cube is, the coarse search only has to comb that region, which is
// both far cheaper than a full-frame scan and immune to structure elsewhere in
// the room.
export function cropFrame(frame, x0, y0, x1, y1) {
  const cx0 = Math.max(0, Math.floor(x0)), cy0 = Math.max(0, Math.floor(y0));
  const cx1 = Math.min(frame.width, Math.ceil(x1)), cy1 = Math.min(frame.height, Math.ceil(y1));
  const w = cx1 - cx0, h = cy1 - cy0;
  if (w < 8 || h < 8) return null;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const src = ((cy0 + y) * frame.width + cx0) * 4;
    out.set(frame.data.subarray(src, src + w * 4), y * w * 4);
  }
  return { data: out, width: w, height: h, offsetX: cx0, offsetY: cy0 };
}

// --- quads and perspective --------------------------------------------------
// A rotated square cannot represent a cube held at an angle: the face projects
// to a trapezoid, with the far edge shorter and the seams unevenly spaced. That
// mismatch is what caps localisation accuracy on real hand-held frames, so the
// scorer works through a homography and the search can move each corner freely.

// corners are [TL, TR, BR, BL] in image pixels
export function quadFromPose(cx, cy, size, theta) {
  const cos = Math.cos(theta), sin = Math.sin(theta), h = size / 2;
  return [[-h, -h], [h, -h], [h, h], [-h, h]]
    .map(([u, v]) => [cx + u * cos - v * sin, cy + u * sin + v * cos]);
}

// Best-fit centre / side / rotation of a quad, for consumers that still think
// in poses (the tracker's pose-consistency test, the overlay transform).
export function poseFromQuad(q) {
  const cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4;
  const cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4;
  const side = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const size = (side(q[0], q[1]) + side(q[1], q[2]) + side(q[2], q[3]) + side(q[3], q[0])) / 4;
  // average the two horizontal edges' directions
  const theta = Math.atan2(
    (q[1][1] - q[0][1]) + (q[2][1] - q[3][1]),
    (q[1][0] - q[0][0]) + (q[2][0] - q[3][0]),
  );
  return { cx, cy, size, theta };
}

// Heckbert's unit-square-to-quad homography. (u,v) in [0,1]^2 -> image.
function homography(q) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, sx = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, sy = y0 - y1 + y2 - y3;
  let a13 = 0, a23 = 0;
  if (sx !== 0 || sy !== 0) {
    const det = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(det) < 1e-9) return null;
    a13 = (sx * dy2 - dx2 * sy) / det;
    a23 = (dx1 * sy - sx * dy1) / det;
  }
  return {
    a11: x1 - x0 + a13 * x1, a21: x3 - x0 + a23 * x3, a31: x0,
    a12: y1 - y0 + a13 * y1, a22: y3 - y0 + a23 * y3, a32: y0,
    a13, a23,
  };
}

// Same interface as windowSampler, so the scorer is unchanged: (u,v) run from
// -size/2 to +size/2 where `size` is the quad's mean side length.
export function quadSampler(frame, q) {
  const H = homography(q);
  const side = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const size = (side(q[0], q[1]) + side(q[1], q[2]) + side(q[2], q[3]) + side(q[3], q[0])) / 4;
  if (!H || size < 1) return null;
  const toImage = (u, v) => {
    const s = u / size + 0.5, t = v / size + 0.5;
    const w = H.a13 * s + H.a23 * t + 1;
    return [(H.a11 * s + H.a21 * t + H.a31) / w, (H.a12 * s + H.a22 * t + H.a32) / w];
  };
  return {
    size,
    px: (u, v) => { const [x, y] = toImage(u, v); return tap(frame, x, y); },
    inFrame: (u, v) => {
      const [x, y] = toImage(u, v);
      return x >= 0 && y >= 0 && x < frame.width && y < frame.height;
    },
  };
}
