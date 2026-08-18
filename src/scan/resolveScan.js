// The scanner's colour resolution step, in one place.
//
// Given the 96 raw RGB samples of a fully scanned cube, cluster them into six
// balanced groups (no reference palette), name those groups by the cube's own
// piece geometry, and report the result in the vocabulary the rest of the app
// speaks: colour letters, per-sticker confidence, and the measured palette for
// display. Both the live cube assembly and the auto-repair search need exactly
// this, so neither reimplements it.
import { clusterStickers } from './cluster.js';
import { labelClusters } from './resolve.js';
import { COLOR_OF_FACE } from '../state/colors.js';

// rgbAll: 96 x [r,g,b] in home-frame facelet order
// -> { colors, alt, conf, palette }, each 96 long except palette (6 css colours)
export function resolveScanColors(rgbAll) {
  const cl = clusterStickers(rgbAll);
  const { faceOfCluster } = labelClusters(cl.assign, cl.centroids);
  const letterOf = (cluster) => COLOR_OF_FACE[faceOfCluster[cluster]];

  const palette = {};
  for (let k = 0; k < 6; k++) {
    const [r, g, b] = cl.centroids[k];
    palette[letterOf(k)] = `rgb(${r}, ${g}, ${b})`;
  }

  return {
    colors: Array.from(cl.assign, letterOf),
    alt: Array.from(cl.alt, letterOf),
    conf: cl.conf,
    palette,
  };
}
