// Sticker color classification from RGB samples (HSV heuristic).
const HUES = { R: 4, O: 27, Y: 52, G: 132, B: 218 };

export function rgb2hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = 60 * (((g - b) / d) % 6);
    else if (mx === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  return [h, mx === 0 ? 0 : d / mx, mx];
}

function hueDist(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// A provisional per-frame read, used only for the live preview and for the
// stability signal. The authoritative colours come from clustering all 96
// samples once every face is captured (see scan/cluster.js).
// -> { key: 'W'|'Y'|'G'|'B'|'R'|'O', conf: 0..1 }
export function classify(r, g, b) {
  const [h, s, v] = rgb2hsv(r, g, b);
  if (v < 0.10) return { key: 'W', conf: 0 };
  const scored = Object.keys(HUES)
    .map((k) => ({ k, d: hueDist(h, HUES[k]) }))
    .sort((a, b2) => a.d - b2.d);
  const whiteScore = Math.max(0, 1 - s / 0.24) * Math.min(1, v / 0.55);
  const chromaScore = Math.min(1, s / 0.38) * Math.min(1, v / 0.30);
  if (whiteScore > chromaScore) return { key: 'W', conf: Math.min(1, whiteScore) };
  const margin = (scored[1].d - scored[0].d) / (scored[1].d + scored[0].d + 1);
  const conf = Math.max(0, Math.min(1, margin * 1.25)) * chromaScore * (1 - whiteScore * 0.5);
  return { key: scored[0].k, conf };
}
