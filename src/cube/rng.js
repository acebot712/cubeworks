// The project's one pseudo-random source. Every caller seeds it explicitly so
// scrambles, search escapes and tests are all reproducible.
export function mulberry32(seed) {
  let a = seed;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// mulberry32 as an integer picker: pick(n) -> 0..n-1
export function intPicker(seed) {
  const rand = mulberry32(seed);
  return (n) => (rand() * n) | 0;
}
