// Minimal SVG plotting primitives, sized for print rather than for a dashboard.
//
// Conventions borrowed from figures that survive peer review: no gridlines
// competing with the data, no legend when a direct label will do, axis ranges
// that include zero unless there is a reason not to, and a colour set that
// still separates when printed greyscale or read by a colourblind reader
// (Okabe-Ito).
export const C = {
  blue: '#0072B2',
  orange: '#E69F00',
  green: '#009E73',
  red: '#D55E00',
  purple: '#CC79A7',
  sky: '#56B4E9',
  yellow: '#F0E442',
  ink: '#111318',
  mid: '#6B7280',
  faint: '#D8DCE3',
  paper: '#FFFFFF',
};

export const linear = (d0, d1, r0, r1) => {
  const s = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  s.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
  s.domain = [d0, d1];
  s.range = [r0, r1];
  return s;
};

export const log = (d0, d1, r0, r1) => {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const s = (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0 || 1)) * (r1 - r0);
  s.domain = [d0, d1];
  s.range = [r0, r1];
  return s;
};

// "Nice" round ticks — 1, 2, 2.5 or 5 times a power of ten.
export function ticks(d0, d1, count = 6) {
  const raw = (d1 - d0) / count;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
  }
  return out;
}

// Ticks for a log axis. Powers of ten are only useful when the range spans
// several; over a narrow range (a loss that moves between 0.06 and 0.08) they
// produce nothing at all, so fall back to linear-nice values placed at their
// log positions.
export function logTicks(d0, d1) {
  const out = [];
  for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= d0 && v <= d1) out.push(+v.toPrecision(12));
    }
  }
  return out.length >= 3 ? out : ticks(d0, d1, 4).filter((v) => v >= d0 && v <= d1);
}

const FONT = '"Space Grotesk", system-ui, sans-serif';
const MONO = '"JetBrains Mono", ui-monospace, monospace';

// A white outline behind text that sits on top of lines, so labels stay legible
// without an opaque box eating the artwork underneath.
const halo = (on) => (on ? {
  stroke: C.paper, strokeWidth: 3.5, strokeLinejoin: 'round', paintOrder: 'stroke',
} : null);

export function Axes({
  x, y, xLabel, yLabel, xTicks, yTicks, fmtX = String, fmtY = String,
  children, pad = 0,
}) {
  const [x0, x1] = x.range;
  const [y0, y1] = y.range;
  const xt = xTicks ?? ticks(...x.domain);
  const yt = yTicks ?? ticks(...y.domain);
  return (
    <g>
      {/* horizontal rules, behind the data and barely there */}
      {yt.map((t) => (
        <line key={`g${t}`} x1={x0} x2={x1} y1={y(t)} y2={y(t)}
          stroke={C.faint} strokeWidth="1" />
      ))}
      <line x1={x0 - pad} x2={x1} y1={y0} y2={y0} stroke={C.ink} strokeWidth="1.25" />
      <line x1={x0 - pad} x2={x0 - pad} y1={y0} y2={y1} stroke={C.ink} strokeWidth="1.25" />
      {xt.map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={y0} y2={y0 + 5} stroke={C.ink} strokeWidth="1.25" />
          <text x={x(t)} y={y0 + 18} textAnchor="middle"
            style={{ font: `11px ${MONO}`, fill: C.mid }}>{fmtX(t)}</text>
        </g>
      ))}
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={x0 - pad - 5} x2={x0 - pad} y1={y(t)} y2={y(t)} stroke={C.ink} strokeWidth="1.25" />
          <text x={x0 - pad - 9} y={y(t) + 4} textAnchor="end"
            style={{ font: `11px ${MONO}`, fill: C.mid }}>{fmtY(t)}</text>
        </g>
      ))}
      {xLabel && (
        <text x={(x0 + x1) / 2} y={y0 + 40} textAnchor="middle"
          style={{ font: `600 12px ${FONT}`, fill: C.ink }}>{xLabel}</text>
      )}
      {yLabel && (
        <text transform={`translate(${x0 - pad - 44} ${(y0 + y1) / 2}) rotate(-90)`}
          textAnchor="middle" style={{ font: `600 12px ${FONT}`, fill: C.ink }}>{yLabel}</text>
      )}
      {children}
    </g>
  );
}

export const Line = ({ pts, stroke = C.blue, width = 2, dash }) => (
  <path fill="none" stroke={stroke} strokeWidth={width} strokeDasharray={dash}
    strokeLinejoin="round" strokeLinecap="round"
    d={pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(2)} ${py.toFixed(2)}`).join('')} />
);

export const Dots = ({ pts, fill = C.blue, r = 3, stroke = C.paper }) => (
  <g>{pts.map(([px, py], i) => (
    <circle key={i} cx={px} cy={py} r={r} fill={fill} stroke={stroke} strokeWidth="1" />
  ))}</g>
);

export const Label = ({ x, y, children, fill = C.ink, anchor = 'start', size = 12, bold, over }) => (
  <text x={x} y={y} textAnchor={anchor} {...halo(over)}
    style={{ font: `${bold ? 600 : 400} ${size}px ${FONT}`, fill }}>{children}</text>
);

export const Mono = ({ x, y, children, fill = C.mid, anchor = 'start', size = 11, over }) => (
  <text x={x} y={y} textAnchor={anchor} {...halo(over)}
    style={{ font: `${size}px ${MONO}`, fill }}>{children}</text>
);

// 10^45 with a real superscript, for axis and value labels.
export const Pow = ({ x, y, base = 10, exp, fill = C.mid, anchor = 'start', size = 11, prefix }) => (
  <text x={x} y={y} textAnchor={anchor} style={{ font: `${size}px ${MONO}`, fill }}>
    {prefix ? `${prefix} ` : ''}{base}
    <tspan dy={-size * 0.38} style={{ fontSize: `${size * 0.75}px` }}>{exp}</tspan>
  </text>
);

// A figure card: numbered, captioned, and downloadable as standalone SVG so the
// thing on screen is the thing you would actually drop into a paper.
export function Figure({ n, title, caption, width = 720, height = 380, children }) {
  const id = `fig-${n}`;
  const download = () => {
    const svg = document.getElementById(id).cloneNode(true);
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const blob = new Blob([
      '<?xml version="1.0" encoding="UTF-8"?>\n',
      new XMLSerializer().serializeToString(svg),
    ], { type: 'image/svg+xml' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `figure-${n}.svg`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <figure className="fig">
      <div className="fig-head">
        <span className="fig-n">Figure {n}</span>
        <h3>{title}</h3>
        <button type="button" onClick={download} title="Download as SVG">SVG</button>
      </div>
      <div className="fig-body">
        <svg id={id} viewBox={`0 0 ${width} ${height}`} width="100%"
          style={{ display: 'block', background: C.paper }}>
          <rect width={width} height={height} fill={C.paper} />
          {children}
        </svg>
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
