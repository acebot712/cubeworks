// 01 Scan — camera stage with the tracking overlay, plus the per-face list.
import React, { useEffect, useState } from 'react';
import { CAPTURE_STEPS } from '../../scan/orientations.js';
import { Btn } from '../primitives.jsx';
import { StickerGrid, ColorPicker, FACE_LIST, faceBase, colorLookup } from '../cubeFaces.jsx';
import { ACCENT, WARN, WARN_DEEP, INFO, INK, INK_SOFT, INK_DIM, INK_MUTED, INK_SUBTLE, INK_FAINT, INK_GHOST, INK_DISABLED, BG_CARD, BG_INERT, mono, sidePanel, sectionLabel, stage, LINE, LINE_SOFT } from '../theme.js';
import { primaryAction, hintFor } from './scanGuidance.js';
import { CONF_WEAK } from '../../scan/assemble.js';
import CameraStage from './CameraStage.jsx';

export default function ScanScreen({ camera, scan, loop, cube, step, stepIdx, awaitingTurn, scanning, labels, actions }) {
  const colorOf = colorLookup(cube.palette);
  const { live } = loop;
  const found = !!(live && live.found);
  const lighting = live ? live.lighting : 0.6;
  const moving = !!(live && live.moving);
  const lowLight = !!live && live.lighting < 0.26 && !camera.error;

  const guidance = { allCaptured: cube.allCaptured, scanning, awaitingTurn, manualQuad: loop.manualQuad, found, lowLight, moving, dwell: loop.dwell, searchMs: loop.searchMs, step };
  const action = primaryAction(guidance, actions);
  const hint = hintFor(guidance);

  const frameColor = loop.manualQuad ? WARN : awaitingTurn ? INFO : loop.dwell > 40 ? ACCENT : 'rgba(255,255,255,0.4)';

  return (
    <div style={stage}>
      <CameraStage
        videoRef={camera.videoRef}
        camera={camera}
        live={live}
        manualQuad={loop.manualQuad}
        setManualQuad={loop.setManualQuad}
        clearManualFrame={loop.clearManualFrame}
        placeManualFrame={actions.placeManualFrame}
        onSampleCube={actions.useSampleCube}
        frameColor={frameColor}
        labels={labels}
        action={action}
        hint={hint}
        lowLight={lowLight}
        showDwell={scanning && found && !awaitingTurn && !cube.allCaptured}
        dwell={loop.dwell}
        lighting={lighting}
      />
      <FacePanel
        cube={cube}
        colorOf={colorOf}
        labels={labels}
        stepIdx={stepIdx}
        step={step}
        scanning={scanning}
        rawCaptures={scan.rawCaptures}
        onSetColor={scan.setManual}
        onRescan={actions.rescanFace}
        onSelectStep={actions.selectStep}
        onReview={actions.goReview}
      />
    </div>
  );
}

// ---------------------------------------------------------------- face list
function FacePanel({ cube, colorOf, labels, stepIdx, step, scanning, rawCaptures, onSetColor, onRescan, onSelectStep, onReview }) {
  const [editFace, setEditFace] = useState(null);
  const [editIdx, setEditIdx] = useState(null);

  // a face can be re-scanned or reset while its editor is open
  useEffect(() => {
    if (editFace && !rawCaptures[editFace]) { setEditFace(null); setEditIdx(null); }
  }, [editFace, rawCaptures]);

  return (
    <div style={sidePanel}>
      <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ ...sectionLabel, marginBottom: 9 }}>GUIDANCE</div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: INK }}>
          {cube.allCaptured ? 'All six faces captured' : `Next up: ${step.instr.toLowerCase()}`}
        </div>
        <div style={{ fontSize: 11, color: INK_GHOST, marginTop: 6, lineHeight: 1.5 }}>
          I follow a six-face loop, but you can click any face below to scan it instead —
          or click a scanned face to fix its colours by hand.
        </div>
      </div>

      <div style={{ padding: '14px 16px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <div style={{ fontSize: 32, fontWeight: 700, letterSpacing: '-0.03em' }}>
            {cube.capturedCount}<span style={{ fontSize: 16, color: INK_GHOST }}>/6</span>
          </div>
          <div style={{ fontSize: 12, color: INK_MUTED }}>faces locked</div>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 10px' }}>
        {FACE_LIST.map((face) => (
          <FaceRow
            key={face.key}
            face={face}
            cube={cube}
            colorOf={colorOf}
            labels={labels}
            done={!!rawCaptures[face.key]}
            isCurrent={CAPTURE_STEPS[stepIdx] && CAPTURE_STEPS[stepIdx].key === face.key}
            scanning={scanning}
            editing={editFace === face.key}
            editIdx={editIdx}
            setEditIdx={setEditIdx}
            onToggleEdit={() => { setEditFace(editFace === face.key ? null : face.key); setEditIdx(null); }}
            onScanNext={() => onSelectStep(CAPTURE_STEPS.findIndex((s) => s.key === face.key))}
            onRescan={() => onRescan(face.key)}
            onSetColor={onSetColor}
          />
        ))}
      </div>

      <div style={{ padding: 14, borderTop: `1px solid ${LINE}` }}>
        {cube.allCaptured ? (
          <div style={{ fontSize: 11.5, color: INK_GHOST, textAlign: 'center', lineHeight: 1.5 }}>
            All six faces are in — use <strong style={{ color: INK_DIM }}>Confirm the read</strong> over the camera,
            or click a face above to fix it first.
          </div>
        ) : (
          <Btn onClick={onReview} style={{ padding: 12, borderRadius: 10, textAlign: 'center', fontSize: 13, fontWeight: 600, color: INK_DIM, border: `1px solid ${LINE_SOFT}` }}>
            Confirm what I have so far
          </Btn>
        )}
      </div>
    </div>
  );
}

function FaceRow({ face, cube, colorOf, labels, done, isCurrent, scanning, editing, editIdx, setEditIdx, onToggleEdit, onScanNext, onRescan, onSetColor }) {
  const base = faceBase(face.key);
  let weakN = 0;
  if (done) for (let i = 0; i < 16; i++) if (cube.conf[base + i] < CONF_WEAK) weakN++;
  const barColor = !done ? BG_INERT : weakN > 3 ? WARN_DEEP : ACCENT;

  let status;
  if (!done) status = isCurrent ? (scanning ? 'Reading now…' : 'Up next — press Start') : 'Waiting';
  else if (editing) status = 'Editing — tap a sticker';
  else status = weakN > 0 ? `${weakN} sticker${weakN > 1 ? 's' : ''} to confirm` : 'Clean read';

  return (
    <div style={{ marginBottom: 4 }}>
      <Btn
        onClick={() => (done ? onToggleEdit() : onScanNext())}
        label={done ? `Edit ${face.name}` : `Scan ${face.name} next`}
        style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 10, borderRadius: 10, background: editing ? 'rgba(127,168,245,0.10)' : isCurrent && !done ? 'rgba(79,227,193,0.07)' : 'transparent' }}
      >
        <div style={{ width: 34, height: 34, flex: 'none', borderRadius: 7, overflow: 'hidden', background: 'rgba(255,255,255,0.06)', opacity: done ? 1 : 0.3 }}>
          <StickerGrid faceKey={face.key} colors={cube.colors} colorOf={colorOf} gap={1} radius={0} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ fontSize: 12.5, fontWeight: 500 }}>{face.name}</div>
            <div style={{ ...mono, fontSize: 9.5, color: INK_FAINT }}>{face.key}</div>
          </div>
          <div style={{ fontSize: 11, color: barColor, marginTop: 3 }}>{status}</div>
        </div>
        <Btn
          onClick={onRescan}
          disabled={!done}
          label={`Re-scan ${face.name}`}
          style={{ flex: 'none', fontSize: 11, padding: '4px 8px', borderRadius: 6, color: done ? INK_SOFT : INK_DISABLED, border: `1px solid ${done ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.05)'}` }}
        >
          {done ? 'Re-scan' : '—'}
        </Btn>
      </Btn>

      {editing && (
        <div style={{ padding: '10px 10px 12px', borderRadius: 10, background: BG_CARD, border: '1px solid rgba(127,168,245,0.22)', marginTop: 4, animation: 'cw-fade .18s ease' }}>
          <StickerGrid
            faceKey={face.key}
            colors={cube.colors}
            colorOf={colorOf}
            labels={labels}
            gap={4}
            radius={5}
            onPick={(idx) => setEditIdx(editIdx === idx ? null : idx)}
            highlight={(idx) => (editIdx === idx ? 'selected' : cube.conf[idx] < CONF_WEAK ? 'weak' : null)}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10 }}>
            <ColorPicker
              colorOf={colorOf}
              current={editIdx == null ? null : cube.colors[editIdx]}
              disabled={editIdx == null}
              onPick={(key) => onSetColor(editIdx, key)}
              size={26}
              stretch
            />
          </div>
          <div style={{ fontSize: 10.5, color: INK_SUBTLE, marginTop: 8 }}>
            {editIdx == null ? 'Tap a sticker, then pick its colour.' : `Sticker ${(editIdx % 16) + 1} — pick its colour.`}
          </div>
        </div>
      )}
    </div>
  );
}
