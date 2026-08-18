// One place for the colours, type and repeated container styles the screens
// share, so a panel or a section label looks the same wherever it appears.
export const ACCENT = '#4FE3C1';
export const ACCENT_INK = '#05201A';
export const WARN = '#FF9E52';
export const WARN_DEEP = '#FF7A1A';
export const BAD = '#E8402A';
export const INFO = '#7FA8F5';

// Text, brightest to faintest.
export const INK = '#E8EAED';
export const INK_SOFT = '#C7CDD4';
export const INK_DIM = '#9AA2AC';
export const INK_MUTED = '#8A929C';
export const INK_SUBTLE = '#7A828C';
// The two faintest tiers that still carry reading text: section labels and the
// hint paragraphs. Both sat under WCAG AA on the panel and card surfaces
// (4.15/3.91 and 3.14/2.96 against a 4.5 floor), so the ramp compressed at the
// bottom rather than staying legible. Raised to clear 4.5 on both; measure
// against BG_PANEL #0E1013 and BG_CARD #14171C before lowering either again.
export const INK_FAINT = '#7C858F';   // 5.09 panel / 4.80 card
export const INK_GHOST = '#79828C';   // 4.89 panel / 4.61 card
export const INK_DISABLED = '#4A5058';

// Surfaces, darkest to lightest.
export const BG = '#08090B';
export const BG_STAGE = '#0A0B0D';
export const BG_PANEL = '#0E1013';
export const BG_CARD = '#14171C';
export const BG_CHIP = '#16191E';
export const BG_CHIP_ON = '#242A31';
export const BG_EMPTY = '#2A2F36';   // an unscanned sticker
export const BG_INERT = '#3A4048';   // an inactive indicator

export const LINE = 'rgba(255,255,255,0.07)';
export const LINE_SOFT = 'rgba(255,255,255,0.1)';
export const LINE_STRONG = 'rgba(255,255,255,0.14)'; // interactive control border

export const mono = { fontFamily: "'JetBrains Mono', monospace" };

// Right-hand panel shared by all three screens.
export const sidePanel = {
  width: 330, flex: 'none', background: BG_PANEL,
  borderLeft: `1px solid ${LINE}`, display: 'flex', flexDirection: 'column',
};

// Small all-caps section heading.
export const sectionLabel = {
  ...mono, fontSize: 9.5, letterSpacing: '0.16em', color: INK_FAINT,
};

export const cardStyle = {
  borderRadius: 11, background: BG_CARD, border: '1px solid rgba(255,255,255,0.08)',
};

// A screen's main stage: fills the space left of the side panel.
export const stage = { flex: 1, display: 'flex', minHeight: 0 };
