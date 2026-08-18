// The camera half of the Scan screen: the preview, the tracking overlay, the
// status pills, the single primary action and the hint bar.
//
// The preview is NOT mirrored. Detection, sampling and the capture maps all
// work in raw camera pixels, so a selfie-style scaleX(-1) would put the one
// thing the user checks their scan against (the live picture) in the opposite
// handedness to the face thumbnails, the editable net and the cube itself.
// Everything on this screen is in raw camera space; there is no mirror to undo.
import React, { useEffect, useRef, useState } from 'react';
import { COLORS } from '../../state/colors.js';
import { Btn, Pill, Meter, Corner } from '../primitives.jsx';
import { ACCENT, ACCENT_INK, WARN, WARN_DEEP, BAD, INK_SOFT, INK_MUTED, INK_SUBTLE, INK_FAINT, INK_DISABLED, BG_STAGE, BG_EMPTY, mono, LINE_STRONG } from '../theme.js';

const FRAME_PX = 300;        // the overlay is drawn at this size, then scaled
const MIN_FRAME = 60;        // smallest hand-placed box, in video px
const ZOOM_STEP = 1.12;

function useStageSize(ref) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

// video pixels -> screen pixels, and screen deltas back again. objectFit:cover
// scales by the larger ratio and centres, so the overflow is split evenly.
function projection(stageSize, vw, vh) {
  if (!stageSize.w) return null;
  const sc = Math.max(stageSize.w / vw, stageSize.h / vh);
  const offX = (vw * sc - stageSize.w) / 2;
  const offY = (vh * sc - stageSize.h) / 2;
  return {
    toScreen: (q) => ({
      cxs: q.cx * sc - offX,
      cys: q.cy * sc - offY,
      sizeS: q.size * sc,
    }),
    toVideoDelta: (dxs, dys) => ({ dx: dxs / sc, dy: dys / sc }),
  };
}

export default function CameraStage({
  videoRef, camera, live, manualQuad, setManualQuad, clearManualFrame, placeManualFrame,
  onSampleCube, frameColor, labels, action, hint, lowLight, showDwell, dwell, lighting,
}) {
  const stageRef = useRef(null);
  const stageSize = useStageSize(stageRef);

  const vw = (live && live.vw) || 1280, vh = (live && live.vh) || 720;
  const proj = projection(stageSize, vw, vh);

  const shownQuad = manualQuad || (live && live.found ? live.quad : null);
  const sp = proj && shownQuad ? proj.toScreen(shownQuad) : null;
  // theta is measured in image space (x right, y down) and CSS rotates the same
  // way, so it carries over with no sign change.
  const overlay = sp && {
    transform: `translate(${sp.cxs - FRAME_PX / 2}px, ${sp.cys - FRAME_PX / 2}px) `
      + `rotate(${shownQuad.theta || 0}rad) scale(${sp.sizeS / FRAME_PX})`,
  };

  const resize = (factor) => setManualQuad((q) => ({ ...q, size: Math.max(MIN_FRAME, q.size * factor) }));

  const onFrameDown = (e) => {
    if (!manualQuad || !proj) return;
    e.preventDefault();
    const origin = { x: e.clientX, y: e.clientY, quad: { ...manualQuad } };
    const move = (ev) => {
      const { dx, dy } = proj.toVideoDelta(ev.clientX - origin.x, ev.clientY - origin.y);
      setManualQuad({ ...origin.quad, cx: origin.quad.cx + dx, cy: origin.quad.cy + dy });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const onFrameWheel = (e) => {
    if (!manualQuad) return;
    const next = Math.max(MIN_FRAME, Math.min(Math.min(vw, vh), manualQuad.size * (e.deltaY > 0 ? 0.94 : 1.06)));
    setManualQuad({ ...manualQuad, size: next });
  };

  return (
    <div ref={stageRef} style={{ flex: 1, position: 'relative', background: BG_STAGE, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
      <video ref={videoRef} autoPlay muted playsInline style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.95 }} />
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at center, rgba(8,9,11,0) 26%, rgba(8,9,11,0.86) 100%)' }} />

      {camera.error && <CameraError camera={camera} onSampleCube={onSampleCube} />}

      {/* placing the frame by hand must not be a one-way door */}
      {manualQuad && !camera.error && (
        <div style={{ position: 'absolute', top: 16, right: 18, display: 'flex', alignItems: 'center', gap: 8, zIndex: 6 }}>
          <Btn onClick={() => resize(ZOOM_STEP)} label="Make the box bigger" style={iconBtn}>+</Btn>
          <Btn onClick={() => resize(1 / ZOOM_STEP)} label="Make the box smaller" style={iconBtn}>−</Btn>
          <Btn onClick={placeManualFrame} style={textBtn}>Recentre</Btn>
          <Btn onClick={clearManualFrame} style={{ ...textBtn, color: WARN, border: '1px solid rgba(255,158,82,0.4)' }}>Back to auto-detect</Btn>
        </div>
      )}

      <div style={{ position: 'absolute', top: 16, left: 18, display: 'flex', alignItems: 'center', gap: 10, zIndex: 3 }}>
        <Pill>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: BAD, animation: 'cw-blink 1.6s ease-in-out infinite' }} />
          <div style={{ ...mono, fontSize: 10.5, letterSpacing: '0.1em' }}>LIVE</div>
        </Pill>
        <Pill>
          <div style={{ ...mono, fontSize: 10.5, color: INK_MUTED }}>LIGHT</div>
          <Meter pct={Math.min(1, lighting * 1.4) * 100} color={lowLight ? WARN_DEEP : ACCENT} />
          <div style={{ ...mono, fontSize: 10.5, color: lowLight ? WARN_DEEP : ACCENT }}>{lowLight ? 'LOW' : 'OK'}</div>
        </Pill>
        <LockPill live={live} />
      </div>

      {/* Follows the detected face, or is dragged by hand. Nothing is drawn at
          all when no face is located. */}
      <div
        onPointerDown={onFrameDown}
        onWheel={onFrameWheel}
        style={overlay
          ? { position: 'absolute', left: 0, top: 0, width: FRAME_PX, height: FRAME_PX, zIndex: 3, transformOrigin: 'center', willChange: 'transform', cursor: manualQuad ? 'grab' : 'default', touchAction: manualQuad ? 'none' : 'auto', ...overlay }
          : { position: 'relative', width: FRAME_PX, height: FRAME_PX, zIndex: 3, opacity: 0, pointerEvents: 'none' }}
      >
        <LiveCells live={live} labels={labels} hidden={!!manualQuad} />
        <Corner style={{ top: -9, left: -9, borderTop: `2px solid ${frameColor}`, borderLeft: `2px solid ${frameColor}`, borderRadius: '4px 0 0 0' }} />
        <Corner style={{ top: -9, right: -9, borderTop: `2px solid ${frameColor}`, borderRight: `2px solid ${frameColor}`, borderRadius: '0 4px 0 0' }} />
        <Corner style={{ bottom: -9, left: -9, borderBottom: `2px solid ${frameColor}`, borderLeft: `2px solid ${frameColor}`, borderRadius: '0 0 0 4px' }} />
        <Corner style={{ bottom: -9, right: -9, borderBottom: `2px solid ${frameColor}`, borderRight: `2px solid ${frameColor}`, borderRadius: '0 0 4px 0' }} />
      </div>

      {action && !camera.error && (
        <Btn
          onClick={action.onClick}
          style={{ position: 'absolute', bottom: 108, left: '50%', transform: 'translateX(-50%)', zIndex: 5, display: 'flex', alignItems: 'center', gap: 10, padding: '14px 28px', borderRadius: 14, background: action.bg, color: action.fg, fontSize: 15, fontWeight: 700, boxShadow: `0 8px 30px ${action.glow}` }}
        >
          <span style={{ fontSize: 13 }}>{action.icon}</span>{action.label}
        </Btn>
      )}

      <div style={{ position: 'absolute', bottom: 22, left: '50%', transform: 'translateX(-50%)', display: 'flex', alignItems: 'center', gap: 13, padding: '12px 16px', borderRadius: 14, background: 'rgba(14,16,19,0.92)', border: `1px solid ${hint.border}`, zIndex: 4, minWidth: 460 }}>
        <div style={{ width: 34, height: 34, flex: 'none', borderRadius: 9, background: hint.chipBg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 15, color: hint.chipFg }}>{hint.icon}</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.01em', color: hint.fg }}>{hint.title}</div>
          <div style={{ fontSize: 11.5, color: INK_MUTED, marginTop: 2 }}>{hint.sub}</div>
        </div>
        {showDwell && (
          <>
            <div style={{ width: 1, height: 30, background: 'rgba(255,255,255,0.1)' }} />
            <div style={{ position: 'relative', width: 70, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.1)', overflow: 'hidden', flex: 'none' }}>
              <div style={{ position: 'absolute', inset: 0, width: `${dwell}%`, background: ACCENT, transition: 'width 100ms linear' }} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// The meter must report the signal the lock gate actually uses, an earlier
// STEADY meter read frame-diff motion while the gate read quad velocity, so it
// showed ~100% steady while refusing to lock.
function LockPill({ live }) {
  const found = !!(live && live.found);
  const moving = !!(live && live.moving);
  const color = found ? (moving ? WARN_DEEP : ACCENT) : INK_DISABLED;
  return (
    <Pill>
      <div style={{ ...mono, fontSize: 10.5, color: INK_MUTED }}>LOCK</div>
      <Meter pct={Math.min(1, (live && live.lock) || 0) * 100} color={color} />
      <div style={{ ...mono, fontSize: 10.5, color: found ? color : INK_FAINT }}>
        {!found ? 'NONE' : moving ? 'MOVING' : 'HELD'}
      </div>
    </Pill>
  );
}

// The 16 cells the sampler is currently reading, laid over the face in the same
// order the sampler reads them: row-major from the picture's top-left.
function LiveCells({ live, labels, hidden }) {
  const cells = (live && live.cells) || new Array(16).fill(null);
  const confs = (live && live.confs) || new Array(16).fill(0);
  const rgbs = live && live.rgbs;
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gridTemplateRows: 'repeat(4, 1fr)', gap: 4, padding: 4, borderRadius: 8 }}>
      {cells.map((key, i) => {
        const conf = confs[i] || 0;
        return (
          <div key={i} style={{ position: 'relative', border: '1px solid rgba(255,255,255,0.16)', borderRadius: 5, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {!hidden && (
              <>
                <div style={{ width: 26, height: 26, borderRadius: '50%', background: rgbs ? `rgb(${rgbs[i][0] | 0}, ${rgbs[i][1] | 0}, ${rgbs[i][2] | 0})` : key ? COLORS[key] : BG_EMPTY, opacity: 0.2 + Math.min(1, conf * 1.2) * 0.8, display: 'flex', alignItems: 'center', justifyContent: 'center', ...mono, fontSize: 11, fontWeight: 700, color: 'rgba(0,0,0,0.55)' }}>
                  {labels && key ? key : ''}
                </div>
                <div style={{ position: 'absolute', inset: 5, borderRadius: '50%', border: `1.5px solid ${conf < 0.4 ? WARN_DEEP : ACCENT}`, opacity: Math.min(1, conf * 1.3) }} />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CameraError({ camera, onSampleCube }) {
  const message = !camera.denied
    ? 'This browser or device is not offering a camera. You can still walk the whole flow with a sample cube.'
    : camera.retries > 1
      ? 'Still blocked. The browser will not re-ask once you have denied it: open the camera icon in the address bar (or Site settings) and set Camera to Allow, then reload this page.'
      : 'Allow camera when the browser asks: scanning reads real sticker colours from the video.';

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14, background: BG_STAGE, zIndex: 6 }}>
      <div style={{ width: 44, height: 44, borderRadius: 12, border: '1px solid rgba(255,255,255,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20, color: INK_FAINT }}>◉</div>
      <div style={{ fontSize: 15, fontWeight: 600 }}>{camera.denied ? 'Camera access blocked' : 'No camera available'}</div>
      <div style={{ fontSize: 12.5, color: INK_SUBTLE, maxWidth: 340, textAlign: 'center', lineHeight: 1.55 }}>{message}</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <Btn onClick={camera.start} style={{ padding: '8px 14px', borderRadius: 8, fontSize: 12.5, border: `1px solid ${LINE_STRONG}` }}>Try camera again</Btn>
        <Btn onClick={onSampleCube} style={{ padding: '8px 14px', borderRadius: 8, fontSize: 12.5, fontWeight: 600, background: ACCENT, color: ACCENT_INK }}>Use a sample cube</Btn>
      </div>
    </div>
  );
}

const iconBtn = { width: 30, height: 30, borderRadius: 8, fontSize: 15, display: 'flex', alignItems: 'center', justifyContent: 'center', color: INK_SOFT, background: 'rgba(14,16,19,0.9)', border: `1px solid ${LINE_STRONG}` };
const textBtn = { padding: '6px 11px', borderRadius: 8, fontSize: 12, color: INK_SOFT, background: 'rgba(14,16,19,0.9)', border: `1px solid ${LINE_STRONG}` };
