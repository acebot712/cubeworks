// Multi-frame accumulation for a face capture (Touch-ID style): several
// consecutive steady frames vote on each cell rather than trusting one.
// Pure, it only sees the frames the sampler produced.

// frames: [{cells, confs, rgbs}] -> { grid, conf, rgb } of 16 entries each
export function voteFrames(frames) {
  const grid = new Array(16);
  const conf = new Array(16);
  const rgb = new Array(16);

  for (let i = 0; i < 16; i++) {
    const votes = new Map();
    let confSum = 0, sr = 0, sg = 0, sb = 0;
    for (const f of frames) {
      votes.set(f.cells[i], (votes.get(f.cells[i]) || 0) + 1);
      confSum += f.confs[i];
      sr += f.rgbs[i][0]; sg += f.rgbs[i][1]; sb += f.rgbs[i][2];
    }
    let winner = null, winnerVotes = 0;
    for (const [key, n] of votes) if (n > winnerVotes) { winner = key; winnerVotes = n; }

    grid[i] = winner;
    // agreement across frames x mean per-frame confidence, nudged up so a clean
    // unanimous read lands comfortably above the review threshold
    conf[i] = Math.min(1, (winnerVotes / frames.length) * (confSum / frames.length) * 1.35);
    rgb[i] = [Math.round(sr / frames.length), Math.round(sg / frames.length), Math.round(sb / frames.length)];
  }

  return { grid, conf, rgb };
}
