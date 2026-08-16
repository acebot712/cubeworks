// Prose for the problem objects validateState reports. Kept out of the
// validator so the domain layer stays free of user-facing copy, and kept out
// of the screens so the wording lives in one place.
import { faceColorName } from '../state/colors.js';

const PHRASE = {
  count: (p) => `${faceColorName(p.face)} appears ${p.count}× (needs 16)`,
  centers: (p) => `${faceColorName(p.face)} centers: ${p.count} of 4`,
  corner: (p) => `impossible corner (${p.colors.map(faceColorName).join('/')})`,
  cornerDup: (p) => `corner ${p.colors} appears ${p.count}×`,
  wing: (p) => `impossible edge sticker pair (${p.colors.map(faceColorName).join('/')})`,
  wingCount: (p) => `edge pair ${p.colors} appears ${p.count}× (needs 2)`,
};

const MAX_SHOWN = 4;

export function describeProblems(problems) {
  return problems.slice(0, MAX_SHOWN).map((p) => PHRASE[p.type] && PHRASE[p.type](p)).filter(Boolean);
}
