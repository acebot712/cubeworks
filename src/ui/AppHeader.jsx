// The stage tabs and the global toggles.
import React from 'react';
import { Btn, ConfirmBtn } from './primitives.jsx';
import { ACCENT, INK_MUTED, INK_FAINT, BG_PANEL, mono, LINE } from './theme.js';

const TABS = [
  { screen: 'scan', num: '01', title: 'Scan' },
  { screen: 'review', num: '02', title: 'Confirm' },
  { screen: 'solve', num: '03', title: 'Solve' },
];

const tabStyle = (active) => ({
  display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px', borderRadius: 7,
  fontSize: 12.5, fontWeight: 500,
  background: active ? 'rgba(79,227,193,0.12)' : 'transparent',
  color: active ? ACCENT : INK_MUTED,
});

export default function AppHeader({ screen, solveReady, labels, onGo, onToggleLabels, onReset }) {
  return (
    <div style={{ height: 52, flex: 'none', display: 'flex', alignItems: 'center', gap: 22, padding: '0 18px', background: BG_PANEL, borderBottom: `1px solid ${LINE}` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
        <div style={{ width: 18, height: 18, borderRadius: 4, background: 'linear-gradient(135deg, #4FE3C1 0%, #2C6BE8 100%)' }} />
        <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.14em' }}>CUBEWORKS</div>
        <div style={{ ...mono, fontSize: 11, color: INK_FAINT, border: '1px solid rgba(255,255,255,0.12)', borderRadius: 4, padding: '2px 6px' }}>4×4×4</div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {TABS.map((tab, i) => {
          const locked = tab.screen === 'solve' && !solveReady;
          return (
            <React.Fragment key={tab.screen}>
              {i > 0 && <div style={{ width: 14, height: 1, background: 'rgba(255,255,255,0.14)' }} />}
              <Btn
                onClick={() => onGo(tab.screen)}
                disabled={locked}
                title={locked ? 'Scan all six faces and fix any problems first' : undefined}
                label={`Step ${Number(tab.num)}: ${tab.title}`}
                style={{ ...tabStyle(screen === tab.screen), opacity: locked ? 0.45 : 1 }}
              >
                <span style={{ ...mono, fontSize: 10, opacity: 0.6 }}>{tab.num}</span>{tab.title}
              </Btn>
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ flex: 1 }} />

      <Btn
        onClick={onToggleLabels}
        label={labels ? 'Hide colour letters' : 'Show colour letters'}
        style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '6px 10px', borderRadius: 7, fontSize: 12, background: labels ? 'rgba(79,227,193,0.12)' : 'transparent', color: labels ? ACCENT : INK_MUTED }}
      >
        <span style={{ ...mono, fontSize: 11 }}>Aa</span>Color labels
      </Btn>
      <ConfirmBtn
        onConfirm={onReset}
        label="Erase the scan and start over"
        style={{ padding: '6px 10px', borderRadius: 7, fontSize: 12, border: '1px solid rgba(255,255,255,0.08)' }}
        armedChildren="Tap again to erase scan"
      >
        Reset
      </ConfirmBtn>
    </div>
  );
}
