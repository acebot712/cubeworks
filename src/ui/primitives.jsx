// Small shared widgets. Clickable surfaces in this app are styled divs;
// Btn gives them the button semantics they need — tab order, Enter/Space and
// an accessible name.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { WARN, INK_MUTED } from './theme.js';

export function Btn({ onClick, style, children, label, disabled, title }) {
  const act = (e) => { if (disabled) return; e.stopPropagation(); onClick && onClick(e); };
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled || undefined}
      aria-label={label}
      title={title}
      onClick={act}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); act(e); }
      }}
      style={{ cursor: disabled ? 'default' : 'pointer', userSelect: 'none', ...style }}
    >
      {children}
    </div>
  );
}

export function Pill({ children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 10px', borderRadius: 20, background: 'rgba(10,11,13,0.72)', border: '1px solid rgba(255,255,255,0.1)' }}>
      {children}
    </div>
  );
}

export function Meter({ pct, color }) {
  return (
    <div style={{ width: 46, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.14)', overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${Math.round(pct)}%`, background: color }} />
    </div>
  );
}

export function Corner({ style }) {
  return <div style={{ position: 'absolute', width: 26, height: 26, ...style }} />;
}

const ARM_MS = 3000;

// A destructive action that asks once. Both places that erase a scan need this,
// and they used to keep their own copy of the arm/disarm timer.
export function ConfirmBtn({ onConfirm, label, style, children, armedChildren }) {
  const [armed, setArmed] = useState(false);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);

  const click = useCallback(() => {
    if (armed) { clearTimeout(timer.current); setArmed(false); onConfirm(); return; }
    setArmed(true);
    timer.current = setTimeout(() => setArmed(false), ARM_MS);
  }, [armed, onConfirm]);

  return (
    <Btn
      onClick={click}
      label={label}
      style={{
        ...style,
        color: armed ? WARN : (style && style.color) || INK_MUTED,
        border: armed ? `1px solid rgba(255,122,26,0.4)` : (style && style.border),
      }}
    >
      {armed ? armedChildren : children}
    </Btn>
  );
}
