// Dataset capture tool — DEV ONLY, deliberately not part of the app's UI.
//
// Point the camera at a cube, click its four corners, save. Or click "No cube
// in frame" to bank a negative — an empty room, curtains, your hands, a
// bookshelf. Negatives matter as much as positives: two of the detector's three
// real-world failures were false positives on background clutter.
//
// Each sample writes three things: a full-resolution JPEG (for humans, and for
// training a model later), the exact downscaled RGB buffer the detector
// consumes (so the eval runner needs no image decoder), and the label.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { detectFace } from '../scan/detect/search.js';
import { createTracker, ACQUIRE_SCORE } from '../scan/detect/tracker.js';
import { DETECT_W } from '../scan/detect/frame.js';

const ACCENT = '#4FE3C1';
const WARN = '#FF9E52';
// Frames are STORED at high resolution and downsampled by whoever consumes
// them, so changing the detector's working resolution never invalidates the
// dataset. The live preview still runs detection at DETECT_W.
const STORE_W = 640;
const mono = { fontFamily: "'JetBrains Mono', monospace" };
const CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];

export default function CaptureApp() {
  const videoRef = useRef(null);
  const wrapRef = useRef(null);
  const detCanvas = useRef(null);
  const fullCanvas = useRef(null);
  const streamRef = useRef(null);
  const trackerRef = useRef(null);

  const [err, setErr] = useState(null);
  const [corners, setCorners] = useState([]);      // normalised [0,1] points
  const [tags, setTags] = useState('');
  const [stats, setStats] = useState({ count: 0, positives: 0, negatives: 0 });
  const [pred, setPred] = useState(null);           // live detector output
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [flash, setFlash] = useState(null);
  const [frozen, setFrozen] = useState(false);

  // ---------- camera ----------
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      .then((s) => {
        streamRef.current = s;
        if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play().catch(() => {}); }
      })
      .catch((e) => setErr(String(e)));
    return () => { if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop()); };
  }, []);

  useEffect(() => {
    fetch('/__capture').then((r) => r.json()).then(setStats).catch(() => {});
  }, []);

  // keep the displayed size so clicks can be normalised
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setBox({ w: r.width, h: r.height });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setBox({ w: r.width, h: r.height });
    return () => ro.disconnect();
  }, []);

  // draw the current video frame at an arbitrary width
  const grabAt = useCallback((width, source) => {
    const v = source || videoRef.current;
    const vw = v.videoWidth || v.naturalWidth, vh = v.videoHeight || v.naturalHeight;
    if (!vw) return null;
    if (!detCanvas.current) detCanvas.current = document.createElement('canvas');
    const c = detCanvas.current;
    const h = Math.max(24, Math.round((width * vh) / vw));
    c.width = width; c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, width, h);
    return { data: ctx.getImageData(0, 0, width, h).data, width, height: h };
  }, []);
  const grabDetFrame = useCallback(() => grabAt(DETECT_W), [grabAt]);

  // RGBA -> base64 of packed RGB
  const packRGB = (f) => {
    const rgb = new Uint8Array(f.width * f.height * 3);
    for (let i = 0, j = 0; i < f.data.length; i += 4, j += 3) {
      rgb[j] = f.data[i]; rgb[j + 1] = f.data[i + 1]; rgb[j + 2] = f.data[i + 2];
    }
    let bin = '';
    for (let i = 0; i < rgb.length; i += 8192) bin += String.fromCharCode(...rgb.subarray(i, i + 8192));
    return btoa(bin);
  };

  // Re-derive every stored frame's buffer from its JPEG at STORE_W. Needed once
  // after raising STORE_W; harmless to repeat.
  const reencode = useCallback(async () => {
    const rows = await fetch('/eval/labels.json').then((r) => r.json()).catch(() => []);
    let done = 0;
    for (const row of rows) {
      const img = new Image();
      img.src = `/eval/frames/${row.id}.jpg`;
      // eslint-disable-next-line no-await-in-loop
      await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; });
      const f = grabAt(STORE_W, img);
      // eslint-disable-next-line no-await-in-loop
      await fetch('/__capture', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ update: row.id, raw: packRGB(f), detW: f.width, detH: f.height }),
      });
      done++;
      setFlash(`Re-derived ${done}/${rows.length} at ${STORE_W}px`);
    }
    setFlash(`Re-derived ${done} frames at ${STORE_W}px`);
    setTimeout(() => setFlash(null), 2500);
  }, [grabAt]);

  // live detector readout, so you can see agreement or disagreement as you label
  useEffect(() => {
    const id = setInterval(() => {
      if (frozen) return;
      const f = grabDetFrame();
      if (!f) return;
      if (!trackerRef.current) trackerRef.current = createTracker();
      const r = trackerRef.current.update(f);
      setPred(r.found && r.quad ? { ...r.quad, score: r.score, w: f.width, h: f.height } : { found: false, score: r.score, w: f.width, h: f.height });
    }, 150);
    return () => clearInterval(id);
  }, [grabDetFrame, frozen]);

  const onClick = (e) => {
    if (corners.length >= 4) return;
    const r = wrapRef.current.getBoundingClientRect();
    setCorners((c) => [...c, [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]]);
  };

  const save = useCallback(async (asNegative) => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const det = grabAt(STORE_W);
    if (!det) return;

    if (!fullCanvas.current) fullCanvas.current = document.createElement('canvas');
    const fc = fullCanvas.current;
    fc.width = v.videoWidth; fc.height = v.videoHeight;
    fc.getContext('2d').drawImage(v, 0, 0);

    const live = trackerRef.current && pred && pred.cx !== undefined
      ? { cx: pred.cx, cy: pred.cy, size: pred.size, theta: pred.theta, score: pred.score }
      : { found: false, score: pred ? pred.score : 0 };

    const res = await fetch('/__capture', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        jpeg: fc.toDataURL('image/jpeg', 0.85),
        raw: packRGB(det),
        corners: asNegative ? null : corners,
        tags,
        detW: det.width, detH: det.height,
        videoW: v.videoWidth, videoH: v.videoHeight,
        predicted: live,
        at: new Date().toISOString(),
      }),
    }).then((r) => r.json());

    if (res.ok) {
      setStats(res);
      setCorners([]);
      setFrozen(false);
      setFlash(asNegative ? 'Saved as a negative (no cube)' : `Saved sample ${res.id}`);
      setTimeout(() => setFlash(null), 1800);
    } else {
      setFlash(`Save failed: ${res.error || 'unknown'}`);
    }
  }, [corners, tags, grabAt, pred]);

  // keyboard: space freezes the frame so a moving cube can be labelled
  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Space') { e.preventDefault(); setFrozen((f) => !f); }
      if (e.key === 'Backspace') { e.preventDefault(); setCorners((c) => c.slice(0, -1)); }
      if (e.key === 'Enter' && corners.length === 4) save(false);
      if (e.key === 'n' || e.key === 'N') save(true);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [corners, save]);

  // project the detector's quad onto the displayed (unmirrored) video
  let predPoly = null;
  if (pred && pred.cx !== undefined && box.w) {
    const sx = box.w / pred.w, sy = box.h / pred.h;
    const c = Math.cos(pred.theta), s = Math.sin(pred.theta), hf = pred.size / 2;
    predPoly = [[-hf, -hf], [hf, -hf], [hf, hf], [-hf, hf]]
      .map(([u, v]) => [(pred.cx + u * c - v * s) * sx, (pred.cy + u * s + v * c) * sy]);
  }

  return (
    <div style={{ minHeight: '100vh', background: '#08090B', color: '#E8EAED', fontFamily: "'Space Grotesk', system-ui, sans-serif", display: 'flex' }}>
      <div style={{ flex: 1, padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.14em' }}>CUBEWORKS CAPTURE</div>
          <div style={{ ...mono, fontSize: 11, color: '#6E7681', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 4, padding: '2px 6px' }}>DEV TOOL</div>
          {frozen && <div style={{ ...mono, fontSize: 11, color: WARN }}>FRAME HELD</div>}
        </div>

        {err && (
          <div style={{ padding: 14, borderRadius: 10, background: 'rgba(232,64,42,0.1)', border: '1px solid rgba(232,64,42,0.3)', marginBottom: 14, fontSize: 13 }}>
            Camera unavailable: {err}
          </div>
        )}

        <div
          ref={wrapRef}
          onClick={onClick}
          style={{ position: 'relative', width: '100%', maxWidth: 960, aspectRatio: '16 / 9', background: '#000', borderRadius: 12, overflow: 'hidden', cursor: corners.length < 4 ? 'crosshair' : 'default' }}
        >
          {/* NOT mirrored: labels must live in the same space the detector sees */}
          <video ref={videoRef} autoPlay muted playsInline style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'fill' }} />

          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            {predPoly && (
              <polygon
                points={predPoly.map((p) => p.join(',')).join(' ')}
                fill="none" stroke={ACCENT} strokeWidth="2" strokeDasharray="6 4" opacity="0.85"
              />
            )}
            {corners.length > 1 && (
              <polygon
                points={corners.map(([x, y]) => `${x * box.w},${y * box.h}`).join(' ')}
                fill="rgba(255,158,82,0.12)" stroke={WARN} strokeWidth="2"
              />
            )}
            {corners.map(([x, y], i) => (
              <g key={i}>
                <circle cx={x * box.w} cy={y * box.h} r="6" fill={WARN} />
                <text x={x * box.w + 10} y={y * box.h - 8} fill={WARN} fontSize="12">{i + 1}</text>
              </g>
            ))}
          </svg>

          <div style={{ position: 'absolute', left: 12, top: 12, display: 'flex', gap: 8 }}>
            <Pill>
              <span style={{ ...mono, fontSize: 10.5, color: '#8A929C' }}>DETECTOR</span>
              <span style={{ ...mono, fontSize: 10.5, color: pred && pred.cx !== undefined ? ACCENT : '#6E7681' }}>
                {pred && pred.cx !== undefined ? `FOUND ${pred.score.toFixed(2)}` : `NONE ${pred ? pred.score.toFixed(2) : '—'}`}
              </span>
            </Pill>
          </div>
        </div>

        <div style={{ marginTop: 12, fontSize: 13, color: '#9AA2AC' }}>
          {corners.length < 4
            ? <>Click the <strong style={{ color: '#E8EAED' }}>{CORNER_NAMES[corners.length]}</strong> corner of the cube face ({corners.length}/4).</>
            : <>Four corners set — press <strong style={{ color: '#E8EAED' }}>Save</strong>, or Backspace to undo.</>}
          {' '}The dashed teal box is what the detector currently thinks.
        </div>
      </div>

      <div style={{ width: 320, flex: 'none', background: '#0E1013', borderLeft: '1px solid rgba(255,255,255,0.07)', padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div>
          <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.16em', color: '#6E7681' }}>DATASET</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 8 }}>
            <div style={{ fontSize: 34, fontWeight: 700 }}>{stats.count}</div>
            <div style={{ fontSize: 12, color: '#8A929C' }}>samples</div>
          </div>
          <div style={{ ...mono, fontSize: 11, color: '#6E7681', marginTop: 4 }}>
            {stats.positives} with a cube · {stats.negatives} without
          </div>
        </div>

        <div>
          <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.16em', color: '#6E7681', marginBottom: 6 }}>CONDITIONS</div>
          <input
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="e.g. white-body, lamp-lit, tilted"
            style={{ width: '100%', padding: '9px 10px', borderRadius: 8, background: '#16191E', border: '1px solid rgba(255,255,255,0.1)', color: '#E8EAED', fontSize: 12.5, fontFamily: 'inherit' }}
          />
          <div style={{ fontSize: 11, color: '#5C636D', marginTop: 6, lineHeight: 1.5 }}>
            Comma-separated. The eval report breaks results down by tag, so this is
            how you find out which conditions actually fail.
          </div>
        </div>

        <Btn onClick={() => save(false)} disabled={corners.length !== 4}
          style={{ background: corners.length === 4 ? ACCENT : '#1A1D22', color: corners.length === 4 ? '#05201A' : '#5C636D' }}>
          Save with corners ⏎
        </Btn>
        <Btn onClick={() => save(true)} style={{ border: `1px solid ${WARN}55`, color: WARN }}>
          No cube in frame — save negative (N)
        </Btn>
        <Btn onClick={() => setCorners([])} style={{ border: '1px solid rgba(255,255,255,0.12)', color: '#9AA2AC' }}>
          Clear corners
        </Btn>
        <Btn onClick={reencode} style={{ border: '1px solid rgba(255,255,255,0.12)', color: '#9AA2AC' }}>
          Re-derive stored frames at {STORE_W}px
        </Btn>
        <Btn onClick={() => setFrozen((f) => !f)} style={{ border: '1px solid rgba(255,255,255,0.12)', color: '#9AA2AC' }}>
          {frozen ? 'Resume detector (space)' : 'Hold frame to label (space)'}
        </Btn>

        {flash && (
          <div style={{ padding: 10, borderRadius: 8, background: 'rgba(79,227,193,0.1)', border: `1px solid ${ACCENT}44`, fontSize: 12, color: '#B6F2E5' }}>{flash}</div>
        )}

        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 11, color: '#5C636D', lineHeight: 1.6 }}>
          Aim for variety over volume: different cubes, rooms, lighting, angles,
          distances — and plenty of negatives from the places it has false-fired.
          Then run <code style={{ ...mono, color: '#9AA2AC' }}>npm run eval</code>.
        </div>
      </div>
    </div>
  );
}

function Pill({ children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 10px', borderRadius: 20, background: 'rgba(10,11,13,0.72)', border: '1px solid rgba(255,255,255,0.1)' }}>
      {children}
    </div>
  );
}

function Btn({ onClick, style, children, disabled }) {
  return (
    <div
      role="button" tabIndex={disabled ? -1 : 0} aria-disabled={disabled || undefined}
      onClick={() => { if (!disabled) onClick(); }}
      onKeyDown={(e) => { if (!disabled && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onClick(); } }}
      style={{ padding: '11px 12px', borderRadius: 9, textAlign: 'center', fontSize: 12.5, fontWeight: 600, cursor: disabled ? 'default' : 'pointer', userSelect: 'none', ...style }}
    >
      {children}
    </div>
  );
}
