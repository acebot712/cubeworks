// 02 Confirm — the whole cube as an editable net, with an opt-in one-at-a-time
// queue for the stickers the scan was unsure about, and the validation panel.
import React, { useMemo, useState } from 'react';
import { FACES } from '../../cube/geometry.js';
import { CENTER_IDX } from '../../cube/pieces.js';
import { COLOR_KEYS, COLOR_OF_FACE, NAMES, NEAR } from '../../state/colors.js';
import { CONF_WEAK } from '../../scan/assemble.js';
import { describeProblems } from '../problemText.js';
import { Btn } from '../primitives.jsx';
import { StickerGrid, ColorPicker, FACE_LIST, FACE_NAME, faceBase, faceOfFacelet, colorLookup } from '../cubeFaces.jsx';
import { ACCENT, ACCENT_INK, WARN, BAD, INFO, INK_DIM, INK_MUTED, INK_SUBTLE, INK_FAINT, INK_GHOST, BG, BG_CHIP, mono, sidePanel, sectionLabel, cardStyle, stage, LINE_SOFT, LINE_STRONG } from '../theme.js';

const NET_POS = { U: [2, 1], L: [1, 2], F: [2, 2], R: [3, 2], B: [4, 2], D: [2, 3] };

// Stickers implicated by validation. A miscounted colour must not light up all
// 16 of its stickers — for a centres problem, narrow to the centre facelets,
// which is what the user can actually fix.
function flaggedStickers(problems, colors) {
  const flagged = new Set();
  for (const p of problems) {
    if (p.stickers) for (const i of p.stickers) flagged.add(i);
    if (p.type === 'centers') {
      const col = COLOR_OF_FACE[p.face];
      for (const f of FACES) for (const i of CENTER_IDX[f]) if (colors[i] === col) flagged.add(i);
    }
  }
  return flagged;
}

function repairSummary(repair) {
  if (!repair || repair.failed || repair.dismissed) return null;
  const parts = [];
  for (const [key, n] of Object.entries(repair.rotations || {})) {
    // n is the correction applied; the user held the face the other way round
    if (n) parts.push(`${FACE_NAME[key]} face was held ${((4 - n) % 4) * 90}° rotated`);
  }
  if (repair.rlSwapped) parts.push('right/left faces swapped');
  const fixed = (repair.fixes || []).length;
  if (fixed) parts.push(`${fixed} sticker${fixed > 1 ? 's' : ''} corrected`);
  if (!parts.length) return null;
  return `Your scan did not describe a real cube, so I corrected it: ${parts.join(', ')}. `
    + 'Corrected stickers are queued for you to confirm.';
}

export default function ReviewScreen({ cube, scan, labels, actions }) {
  const colorOf = colorLookup(cube.palette);
  const [showAll, setShowAll] = useState(true);
  const [queueIdx, setQueueIdx] = useState(0);
  const [selected, setSelected] = useState(null);
  const [skipped, setSkipped] = useState(() => new Set());

  // the stickers the scan is unsure about, least confident first
  const queue = useMemo(() => {
    const q = [];
    for (let i = 0; i < 96; i++) {
      if (cube.colors[i] && cube.conf[i] < CONF_WEAK && scan.manualColors[i] === undefined && !skipped.has(i)) {
        q.push({ idx: i, conf: cube.conf[i] });
      }
    }
    return q.sort((a, b) => a.conf - b.conf);
  }, [cube.colors, cube.conf, scan.manualColors, skipped]);

  const current = queue[queueIdx];
  const flagged = flaggedStickers(cube.validation.problems || [], cube.colors);

  // Answering removes this sticker from the derived queue and shifts the rest
  // down, so queueIdx must stay put — incrementing it too would skip every
  // other flagged sticker.
  const answer = (idx, color) => { scan.setManual(idx, color); actions.invalidateSolution(); };

  return (
    <div className="cw-stage" style={stage}>
      <h1 className="cw-sr-only">Step 2 of 3: confirm the scanned colours</h1>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 24, padding: 24, overflow: 'auto', background: 'radial-gradient(ellipse at 50% 0%, #101318 0%, #08090B 70%)' }}>
        {current && !showAll ? (
          <QueueCard
            entry={current}
            total={queue.length}
            position={queueIdx + 1}
            cube={cube}
            colorOf={colorOf}
            labels={labels}
            onAnswer={answer}
            onSkip={() => setSkipped((prev) => new Set(prev).add(current.idx))}
            onShowAll={() => setShowAll(true)}
          />
        ) : (
          <CubeNet
            cube={cube}
            colorOf={colorOf}
            labels={labels}
            flagged={flagged}
            selected={selected}
            setSelected={setSelected}
            weakTotal={queue.length}
            onReviewOneByOne={() => { setShowAll(false); setQueueIdx(0); }}
            onSetColor={answer}
          />
        )}
      </div>

      <ValidationPanel
        cube={cube}
        colorOf={colorOf}
        labels={labels}
        repair={scan.repair}
        flagged={flagged}
        actions={actions}
      />
    </div>
  );
}

// ------------------------------------------------------------- queue mode
function QueueCard({ entry, total, position, cube, colorOf, labels, onAnswer, onSkip, onShowAll }) {
  const read = cube.colors[entry.idx];
  const options = [
    { key: read, note: `MY READ · ${Math.round(entry.conf * 100)}%` },
    { key: NEAR[read] || 'W', note: 'LIKELY MIX-UP' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, animation: 'cw-fade .25s ease' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ ...mono, fontSize: 10, letterSpacing: '0.16em', color: INK_FAINT }}>STICKER {position} OF {total}</div>
        <div style={{ fontSize: 21, fontWeight: 600, letterSpacing: '-0.02em', marginTop: 8 }}>Which color is this?</div>
        <div style={{ fontSize: 12.5, color: INK_MUTED, marginTop: 5 }}>Only the stickers I'm unsure about — everything else is settled.</div>
      </div>

      <div style={{ position: 'relative', padding: 6, borderRadius: 10, ...cardStyle }}>
        <StickerGrid
          faceKey={faceOfFacelet(entry.idx)}
          colors={cube.colors}
          colorOf={colorOf}
          labels={labels}
          cell={46}
          highlight={(idx) => (idx === entry.idx ? 'focus' : null)}
        />
        <FaceTag>{faceOfFacelet(entry.idx)} FACE</FaceTag>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {options.map((o, i) => (
          <Btn key={i} onClick={() => onAnswer(entry.idx, o.key)} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 18px', borderRadius: 12, background: BG_CHIP, border: `1px solid ${LINE_SOFT}` }}>
            <div style={{ width: 30, height: 30, borderRadius: 7, background: colorOf(o.key), border: '1px solid rgba(255,255,255,0.16)' }} />
            <div>
              <div style={{ fontSize: 14, fontWeight: 600 }}>{NAMES[o.key]}</div>
              <div style={{ ...mono, fontSize: 10, color: INK_SUBTLE, marginTop: 1 }}>{o.note}</div>
            </div>
          </Btn>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 14, fontSize: 12 }}>
        <Btn onClick={onSkip} style={{ color: INK_SUBTLE }}>Leave this one as is</Btn>
        <Btn onClick={onShowAll} style={{ color: ACCENT }}>Show the whole cube instead</Btn>
      </div>
    </div>
  );
}

// --------------------------------------------------------------- net mode
function CubeNet({ cube, colorOf, labels, flagged, selected, setSelected, weakTotal, onReviewOneByOne, onSetColor }) {
  const highlight = (idx) => (selected === idx ? 'selected' : flagged.has(idx) ? 'flagged' : null);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, animation: 'cw-fade .25s ease' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: '-0.02em' }}>
          {weakTotal === 0 && cube.complete ? 'Every uncertain sticker confirmed' : 'Captured state'}
        </div>
        <div style={{ fontSize: 12.5, color: INK_MUTED, marginTop: 5 }}>
          Tap any sticker to change it. Flagged stickers pulse red.
        </div>
        {weakTotal > 0 && (
          <Btn onClick={onReviewOneByOne} style={{ display: 'inline-block', marginTop: 10, padding: '6px 12px', borderRadius: 8, fontSize: 12, color: ACCENT, border: `1px solid ${ACCENT}55` }}>
            Review {weakTotal} uncertain sticker{weakTotal > 1 ? 's' : ''} one by one
          </Btn>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 112px)', gridTemplateRows: 'repeat(3, 112px)', gap: 8 }}>
        {FACE_LIST.map((f) => (
          <div key={f.key} style={{ gridColumn: NET_POS[f.key][0], gridRow: NET_POS[f.key][1], position: 'relative', padding: 5, borderRadius: 8, ...cardStyle }}>
            <StickerGrid
              faceKey={f.key}
              colors={cube.colors}
              colorOf={colorOf}
              labels={labels}
              radius={3}
              onPick={setSelected}
              highlight={highlight}
            />
            <FaceTag>{f.key}</FaceTag>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderRadius: 12, ...cardStyle }}>
        <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.14em', color: INK_FAINT }}>SET TO</div>
        <ColorPicker
          colorOf={colorOf}
          current={selected == null ? null : cube.colors[selected]}
          disabled={selected == null}
          onPick={(key) => onSetColor(selected, key)}
        />
        <div style={{ width: 1, height: 22, background: 'rgba(255,255,255,0.1)' }} />
        <div style={{ fontSize: 11.5, color: selected != null ? INK_MUTED : INK_FAINT }}>
          {selected != null
            ? `${faceOfFacelet(selected)} · sticker ${(selected % 16) + 1} — pick its colour`
            : 'Tap a sticker in the net first'}
        </div>
      </div>
    </div>
  );
}

function FaceTag({ children }) {
  return (
    <div style={{ position: 'absolute', top: -8, left: 6, ...mono, fontSize: 9, letterSpacing: '0.14em', color: INK_FAINT, background: BG, padding: '0 4px' }}>
      {children}
    </div>
  );
}

// ------------------------------------------------------------- side panel
function ValidationPanel({ cube, colorOf, labels, repair, flagged, actions }) {
  const { ok, incomplete, problems } = cube.validation;
  const summary = repairSummary(repair);

  const missingFaces = FACE_LIST
    .filter((f) => {
      const base = faceBase(f.key);
      for (let i = base; i < base + 16; i++) if (!cube.colors[i]) return true;
      return false;
    })
    .map((f) => f.key);

  const flaggedFaces = [...new Set([...flagged].map(faceOfFacelet))];

  let explanation;
  if (ok) explanation = 'Every piece checks out — corners, edge wings and centers are all real. Solver is ready.';
  else if (incomplete) explanation = `${missingFaces.length} face${missingFaces.length > 1 ? 's' : ''} still to scan: ${missingFaces.join(', ')}. Grey squares in the net are unscanned — you can also fill them in by hand.`;
  else explanation = `${describeProblems(problems).join(' · ')}. The stickers involved are pulsing in the net — tap one, then pick its real colour.`;

  return (
    <div className="cw-panel" style={{ ...sidePanel, padding: '18px 16px' }}>
      <h2 style={{ ...sectionLabel, margin: 0 }}>STATE VALIDATION</h2>
      <div style={{ marginTop: 12, padding: 14, borderRadius: 11, background: ok ? 'rgba(79,227,193,0.08)' : 'rgba(232,64,42,0.1)', border: `1px solid ${ok ? 'rgba(79,227,193,0.28)' : 'rgba(232,64,42,0.32)'}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 18, height: 18, borderRadius: '50%', background: ok ? ACCENT : BAD, color: BG, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{ok ? '✓' : '!'}</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: ok ? ACCENT : BAD }}>
            {ok ? 'Valid cube state' : incomplete ? 'Scan incomplete' : 'Impossible cube state'}
          </div>
        </div>
        <div style={{ fontSize: 11.5, color: INK_DIM, marginTop: 8, lineHeight: 1.55 }}>{explanation}</div>

        {/* every failure state needs a forward action */}
        {!ok && (
          <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(incomplete ? missingFaces : flaggedFaces).map((key) => (
              <Btn
                key={key}
                onClick={() => actions.rescanFace(key)}
                label={`${incomplete ? 'Scan' : 'Re-scan'} the ${FACE_NAME[key]} face`}
                style={{ fontSize: 11, padding: '4px 9px', borderRadius: 7, ...(incomplete ? { color: ACCENT, border: `1px solid ${ACCENT}55` } : { color: '#FFC08A', border: '1px solid rgba(255,158,82,0.35)' }) }}
              >
                {incomplete ? 'Scan' : 'Re-scan'} {FACE_NAME[key]}
              </Btn>
            ))}
          </div>
        )}
      </div>

      {/* only while it is still true: a failed repair says nothing about a cube
          the user has since fixed by hand */}
      {repair && repair.failed && !ok && (
        <div style={{ marginTop: 12, padding: 12, borderRadius: 11, background: 'rgba(255,122,26,0.09)', border: '1px solid rgba(255,158,82,0.3)' }}>
          <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.14em', color: WARN }}>AUTO-REPAIR FAILED</div>
          <div style={{ fontSize: 11.5, color: '#D8B79A', marginTop: 7, lineHeight: 1.55 }}>
            I tried every face rotation and a few sticker swaps and still could not make this a real cube,
            so more than a couple of stickers are misread. Re-scanning a flagged face above is usually
            faster than fixing them one by one.
          </div>
        </div>
      )}

      {repair && repair.dismissed && (
        <div style={{ marginTop: 12, padding: 12, borderRadius: 11, ...cardStyle, display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 11.5, color: INK_DIM, flex: 1, lineHeight: 1.5 }}>Auto-repair is off — you are seeing the raw scan.</div>
          <Btn onClick={actions.reapplyRepair} style={{ fontSize: 11, color: INFO, border: '1px solid rgba(127,168,245,0.4)', borderRadius: 6, padding: '3px 8px' }}>Re-apply</Btn>
        </div>
      )}

      {summary && (
        <div style={{ marginTop: 12, padding: 12, borderRadius: 11, background: 'rgba(127,168,245,0.09)', border: '1px solid rgba(127,168,245,0.28)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.14em', color: INFO }}>AUTO-REPAIR</div>
            <div style={{ flex: 1 }} />
            <Btn onClick={actions.undoRepair} style={{ fontSize: 11, color: INK_DIM, border: `1px solid ${LINE_STRONG}`, borderRadius: 6, padding: '3px 8px' }}>Undo</Btn>
          </div>
          <div style={{ fontSize: 11.5, color: '#B9CBEA', marginTop: 7, lineHeight: 1.55 }}>{summary}</div>
        </div>
      )}

      <h2 style={{ ...sectionLabel, margin: '20px 0 0' }}>STICKER COUNT · 16 EACH</h2>
      <div style={{ marginTop: 11, display: 'flex', flexDirection: 'column', gap: 7 }}>
        {COLOR_KEYS.map((key) => {
          const n = cube.counts[key];
          return (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 16, height: 16, borderRadius: 4, background: colorOf(key), border: '1px solid rgba(255,255,255,0.16)', display: 'flex', alignItems: 'center', justifyContent: 'center', ...mono, fontSize: 9, fontWeight: 700, color: 'rgba(0,0,0,0.5)' }}>{labels ? key : ''}</div>
              <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (n / 16) * 100)}%`, background: n === 16 ? ACCENT : BAD }} />
              </div>
              <div style={{ ...mono, fontSize: 10.5, color: n === 16 ? ACCENT : BAD, width: 38, textAlign: 'right' }}>{n}/16</div>
            </div>
          );
        })}
      </div>

      <div style={{ flex: 1 }} />
      <Btn onClick={actions.goScan} style={{ padding: 11, borderRadius: 10, textAlign: 'center', fontSize: 12.5, color: INK_DIM, border: `1px solid ${LINE_SOFT}`, marginBottom: 8 }}>Back to scanning</Btn>
      <Btn
        onClick={actions.requestSolve}
        disabled={!ok}
        style={{ padding: 13, borderRadius: 10, textAlign: 'center', fontSize: 13, fontWeight: 600, background: ok ? ACCENT : '#1A1D22', color: ok ? ACCENT_INK : INK_GHOST }}
      >
        {ok ? 'Solve — reduction method'
          : incomplete ? `Scan ${missingFaces.length} more face${missingFaces.length > 1 ? 's' : ''} to solve`
          : 'Fix the pulsing stickers to solve'}
      </Btn>
    </div>
  );
}
