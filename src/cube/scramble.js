// Random-move scramble generator (WCA-style move set for 4x4).
const BASES = ['U', 'D', 'F', 'B', 'R', 'L', 'Uw', 'Dw', 'Fw', 'Bw', 'Rw', 'Lw'];
const SUFFIX = ['', "'", '2'];

export function randomScramble(len = 45, rand = Math.random) {
  const out = [];
  let prevAxis = -1;
  let prevBase = '';
  const axisOf = (b) => 'UD'.includes(b[0]) ? 0 : 'RL'.includes(b[0]) ? 1 : 2;
  while (out.length < len) {
    const base = BASES[(rand() * BASES.length) | 0];
    if (base === prevBase) continue;
    const ax = axisOf(base);
    if (ax === prevAxis && base[0] === prevBase[0]) continue; // avoid U then Uw etc.
    out.push(base + SUFFIX[(rand() * 3) | 0]);
    prevBase = base;
    prevAxis = ax;
  }
  return out;
}
