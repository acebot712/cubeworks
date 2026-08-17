// 03 Solve — the animated 3D guide, the move transport, and the phase list.
import React, { useMemo } from 'react';
import Cube3D from '../Cube3D.jsx';
import { Btn, ConfirmBtn } from '../primitives.jsx';
import { describeMove, arrowFor } from '../moveText.js';
import { ACCENT, ACCENT_INK, WARN, INK, INK_SOFT, INK_DIM, INK_MUTED, INK_SUBTLE, INK_FAINT, INK_GHOST, INK_DISABLED, BG_CHIP, BG_CHIP_ON, BG_INERT, mono, sidePanel, sectionLabel, stage, LINE, LINE_STRONG } from '../theme.js';

const STRIP_BEFORE = 3;
const STRIP_AFTER = 4;

export default function SolveScreen({ videoRef, solver, playback, states, labels, colorOf, labelOf, actions }) {
  // The camera stays mounted so its stream survives a return to Scan.
  const keepAlive = <video ref={videoRef} autoPlay muted playsInline style={{ display: 'none' }} />;

  if (solver.solving || (!solver.solution && !solver.error)) {
    return <Centered>{keepAlive}<Solving onCancel={actions.goReview} /></Centered>;
  }
  if (solver.error) {
    return <Centered>{keepAlive}<SolveFailed message={solver.error} onRetry={actions.retrySolve} onBack={actions.goReview} /></Centered>;
  }

  return <SolutionView {...{ keepAlive, solver, playback, states, labels, colorOf, labelOf, actions }} />;
}

function Centered({ children }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
      {children}
    </div>
  );
}

function Solving({ onCancel }) {
  return (
    <>
      <div style={{ width: 42, height: 42, border: '3px solid rgba(79,227,193,0.2)', borderTopColor: ACCENT, borderRadius: '50%', animation: 'cw-spin 0.9s linear infinite' }} />
      <div style={{ fontSize: 15, fontWeight: 600 }}>Solving your cube…</div>
      <div style={{ fontSize: 12.5, color: INK_MUTED, maxWidth: 340, textAlign: 'center', lineHeight: 1.6 }}>
        Centers first, then edge pairing, then the 3×3 finish. The whole route is verified move-by-move before you see it.
      </div>
      <Btn onClick={onCancel} style={{ padding: '8px 14px', borderRadius: 9, fontSize: 12.5, color: INK_DIM, border: `1px solid ${LINE_STRONG}` }}>Cancel</Btn>
    </>
  );
}

function SolveFailed({ message, onRetry, onBack }) {
  return (
    <>
      <div style={{ fontSize: 15, fontWeight: 600, color: WARN }}>I couldn't solve that state</div>
      <div style={{ fontSize: 12.5, color: INK_MUTED, maxWidth: 380, textAlign: 'center', lineHeight: 1.6 }}>{message}</div>
      <div style={{ fontSize: 12.5, color: INK_MUTED, maxWidth: 380, textAlign: 'center', lineHeight: 1.6 }}>
        This almost always means a sticker was misread. Go back and double-check the flagged ones.
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <Btn onClick={onRetry} style={{ padding: '9px 16px', borderRadius: 9, fontSize: 12.5, fontWeight: 600, background: ACCENT, color: ACCENT_INK }}>Try again</Btn>
        <Btn onClick={onBack} style={{ padding: '9px 16px', borderRadius: 9, fontSize: 12.5, color: INK_SOFT, border: `1px solid ${LINE_STRONG}` }}>Back to Confirm</Btn>
      </div>
    </>
  );
}

function SolutionView({ keepAlive, solver, playback, states, labels, colorOf, labelOf, actions }) {
  const { flat, phases } = solver.solution;
  const total = flat.length;
  const { moveIdx, playing, setPlaying, setMoveIdx, speed, setSpeed, step, restart } = playback;

  const done = moveIdx >= total;
  const token = done ? null : flat[moveIdx];

  // which phase each move belongs to, and where each phase starts
  const { phaseOfMove, blocks } = useMemo(() => {
    const names = [];
    let start = 0;
    const b = phases.map((p) => {
      const block = { ...p, start };
      for (const _ of p.moves) names.push(p.name);
      start += p.moves.length;
      return block;
    });
    return { phaseOfMove: names, blocks: b };
  }, [phases]);

  const strip = [];
  for (let i = moveIdx - STRIP_BEFORE; i <= moveIdx + STRIP_AFTER; i++) {
    if (i >= 0 && i < total) strip.push(i);
  }

  return (
    <div className="cw-stage" style={stage}>
      <h1 className="cw-sr-only">Step 3 of 3: follow the solution</h1>
      <div className="cw-solve-stage" style={{ flex: 1, position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'radial-gradient(ellipse at 50% 12%, #12161C 0%, #08090B 68%)', overflow: 'hidden' }}>
        {keepAlive}

        <div className="cw-solve-tophint" style={{ position: 'absolute', top: 16, left: 20, display: 'flex', alignItems: 'center', gap: 12, zIndex: 4 }}>
          <GripHint />
          <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.16em', color: ACCENT }}>
            {(done ? 'Done' : phaseOfMove[moveIdx] || phaseOfMove[0] || '').toUpperCase()}
          </div>
        </div>

        <div className="cw-solve-cube">
          <Cube3D states={states || []} moves={flat} idx={moveIdx} colorOf={colorOf} labelOf={labelOf} labels={labels} size={290} />
        </div>

        <div className="cw-solve-card" style={{ position: 'relative', zIndex: 3, marginTop: 30, display: 'flex', alignItems: 'center', gap: 18, padding: '13px 22px', borderRadius: 16, background: 'rgba(14,16,19,0.92)', border: '1px solid rgba(255,255,255,0.09)' }}>
          {done ? (
            <>
              <div style={{ fontSize: 38, color: ACCENT }}>✓</div>
              <div>
                <div style={{ ...mono, fontSize: 30, fontWeight: 700 }}>Solved</div>
                <div style={{ fontSize: 13, color: INK_DIM, marginTop: 3 }}>All {total} moves done — enjoy the moment.</div>
              </div>
              <div style={{ display: 'flex', gap: 8, marginLeft: 6 }}>
                <Btn onClick={restart} style={{ padding: '8px 13px', borderRadius: 9, fontSize: 12.5, color: INK_SOFT, border: '1px solid rgba(255,255,255,0.16)' }}>Watch again</Btn>
                <Btn onClick={actions.startOver} style={{ padding: '8px 13px', borderRadius: 9, fontSize: 12.5, fontWeight: 600, background: ACCENT, color: ACCENT_INK }}>Scan another cube</Btn>
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 38, color: ACCENT, animation: 'cw-nudge 1.4s ease-in-out infinite' }}>{arrowFor(token)}</div>
              <div>
                <div style={{ ...mono, fontSize: 30, fontWeight: 700 }}>{token}</div>
                <div style={{ fontSize: 13, color: INK_DIM, marginTop: 3 }}>{describeMove(token)}</div>
              </div>
            </>
          )}
        </div>

        <div className="cw-solve-chips" style={{ position: 'relative', zIndex: 3, marginTop: 20, display: 'flex', alignItems: 'center', gap: 7 }}>
          {strip.map((i) => (
            <MoveChip key={i} token={flat[i]} state={i === moveIdx ? 'current' : i < moveIdx ? 'past' : 'future'} onClick={() => setMoveIdx(i)} wide />
          ))}
        </div>

        <div className="cw-solve-transport" style={{ position: 'absolute', bottom: 18, left: '50%', transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 9, zIndex: 4 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Btn onClick={() => step(-1)} label="Previous move" className="cw-transport-btn" style={transportBtn}>◀</Btn>
            <Btn onClick={() => setPlaying(!playing)} style={{ height: 44, padding: '0 22px', borderRadius: 12, display: 'flex', alignItems: 'center', gap: 9, fontSize: 13, fontWeight: 600, background: playing ? BG_CHIP_ON : ACCENT, color: playing ? INK : ACCENT_INK }}>
              {playing ? '❙❙  Pause' : '▶  Play along'}
            </Btn>
            <Btn onClick={() => step(1)} label="Next move" className="cw-transport-btn" style={transportBtn}>▶</Btn>
          </div>
          <div className="cw-kbd-hint" style={{ ...mono, fontSize: 9.5, letterSpacing: '0.1em', color: INK_GHOST }}>SPACE PLAY · ← → STEP</div>
        </div>
      </div>

      <div className="cw-panel" style={sidePanel}>
        <div style={{ padding: 16, borderBottom: `1px solid ${LINE}` }}>
          <h2 style={{ ...sectionLabel, margin: 0 }}>SOLUTION</h2>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 7 }}>
            <div style={{ fontSize: 32, fontWeight: 700, letterSpacing: '-0.03em' }}>{total}</div>
            <div style={{ fontSize: 12, color: INK_MUTED }}>moves · reduction method</div>
          </div>
          <div style={{ marginTop: 11, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.09)', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${Math.round((moveIdx / total) * 100)}%`, background: ACCENT, transition: 'width 250ms ease' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 7, ...mono, fontSize: 10, color: INK_FAINT }}>
            <div>MOVE {Math.min(moveIdx + 1, total)} / {total}</div>
            <div>{total - moveIdx} LEFT</div>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
          {blocks.map((p) => (
            <div key={p.name} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: !done && phaseOfMove[moveIdx] === p.name ? ACCENT : BG_INERT }} />
                <div style={{ fontSize: 12, fontWeight: 600, color: !done && phaseOfMove[moveIdx] === p.name ? INK : INK_SUBTLE }}>{p.name}</div>
                <div style={{ ...mono, fontSize: 10, color: INK_GHOST }}>{p.moves.length} moves</div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, paddingLeft: 14 }}>
                {p.moves.map((m, j) => {
                  const gi = p.start + j;
                  return <MoveChip key={j} token={m} state={gi === moveIdx ? 'current' : gi < moveIdx ? 'past' : 'future'} onClick={() => setMoveIdx(gi)} />;
                })}
              </div>
            </div>
          ))}
        </div>

        <div style={{ padding: 14, borderTop: `1px solid ${LINE}`, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: INK_MUTED }}>
            <div>Playback speed</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {[0.5, 1, 2].map((v) => (
                <Btn key={v} onClick={() => setSpeed(v)} label={`Playback speed ${v} times`} style={{ padding: '4px 9px', borderRadius: 6, ...mono, fontSize: 11, background: speed === v ? BG_CHIP_ON : 'transparent', color: speed === v ? INK : INK_SUBTLE }}>{v}×</Btn>
              ))}
            </div>
          </div>
          <ConfirmBtn
            onConfirm={actions.startOver}
            label="Erase this scan and start a new one"
            style={{ padding: 10, borderRadius: 9, textAlign: 'center', fontSize: 12.5, color: WARN, border: '1px solid rgba(255,122,26,0.3)' }}
            armedChildren="Tap again — this erases all six faces"
          >
            Lost track? Re-scan the cube
          </ConfirmBtn>
        </div>
      </div>
    </div>
  );
}

function MoveChip({ token, state, onClick, wide }) {
  const style = wide
    ? { minWidth: 42, padding: 8, fontSize: 13.5, borderRadius: 8 }
    : { minWidth: 34, padding: '5px 6px', fontSize: 11.5, borderRadius: 6 };
  const tone = {
    current: { background: ACCENT, color: ACCENT_INK, border: `1px solid ${ACCENT}`, opacity: 1 },
    past: { background: wide ? 'transparent' : 'rgba(255,255,255,0.04)', color: INK_DISABLED, border: '1px solid rgba(255,255,255,0.08)', opacity: wide ? 0.5 : 1 },
    future: { background: BG_CHIP, color: INK_SOFT, border: '1px solid rgba(255,255,255,0.08)', opacity: wide ? 0.9 : 1 },
  }[state];
  return <Btn onClick={onClick} style={{ ...style, ...tone, textAlign: 'center', ...mono }}>{token}</Btn>;
}

function GripHint() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '8px 12px', borderRadius: 10, background: 'rgba(14,16,19,0.9)', border: '1px solid rgba(255,255,255,0.09)' }}>
      <div style={{ position: 'relative', width: 26, height: 26 }}>
        <div style={{ position: 'absolute', left: 0, top: 6, width: 20, height: 20, borderRadius: 3, background: '#23B15A' }} />
        <div style={{ position: 'absolute', left: 5, top: 0, width: 20, height: 8, borderRadius: 2, background: '#F2F3F5', transform: 'skewX(-38deg)' }} />
      </div>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600 }}>Hold it like this: green front, white up</div>
        <div style={{ fontSize: 10.5, color: INK_MUTED }}>Every move below is relative to this grip</div>
      </div>
    </div>
  );
}

const transportBtn = { width: 40, height: 40, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, background: BG_CHIP, border: '1px solid rgba(255,255,255,0.09)', color: INK_SOFT };
