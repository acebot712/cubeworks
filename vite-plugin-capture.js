// Dev-only endpoint that writes labelled camera frames to eval/ on disk.
//
// The detector's whole test suite is synthetic — images generated from the same
// assumptions the detector encodes, which is circular and has missed every
// real-world failure so far. This lets real frames from a real camera become
// the benchmark instead.
import fs from 'node:fs';
import path from 'node:path';

const DIR = 'eval';
const FRAMES = path.join(DIR, 'frames');
const LABELS = path.join(DIR, 'labels.json');

function readLabels() {
  try { return JSON.parse(fs.readFileSync(LABELS, 'utf8')); } catch { return []; }
}

export default function capturePlugin() {
  return {
    name: 'cubeworks-capture',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__capture', (req, res) => {
        if (req.method === 'GET') {
          const labels = readLabels();
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify({
            count: labels.length,
            positives: labels.filter((l) => l.corners).length,
            negatives: labels.filter((l) => !l.corners).length,
          }));
          return;
        }
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        let body = '';
        req.on('data', (c) => { body += c; });
        req.on('end', () => {
          try {
            const s = JSON.parse(body);
            fs.mkdirSync(FRAMES, { recursive: true });
            const labels = readLabels();

            // re-derive an existing sample's buffer at a new resolution, so the
            // dataset is not locked to whatever DETECT_W was current when it
            // was captured
            if (s.update) {
              const row = labels.find((l) => l.id === s.update);
              if (!row) { res.statusCode = 404; res.end(JSON.stringify({ error: 'no such sample' })); return; }
              fs.writeFileSync(path.join(FRAMES, `${s.update}.bin`), Buffer.from(s.raw, 'base64'));
              row.detW = s.detW; row.detH = s.detH;
              fs.writeFileSync(LABELS, `${JSON.stringify(labels, null, 1)}\n`);
              res.setHeader('content-type', 'application/json');
              res.end(JSON.stringify({ ok: true, id: s.update, updated: true }));
              return;
            }

            const id = String(labels.length + 1).padStart(4, '0');

            // full-resolution JPEG: for human inspection and, later, training
            fs.writeFileSync(
              path.join(FRAMES, `${id}.jpg`),
              Buffer.from(s.jpeg.split(',')[1], 'base64'),
            );
            // the exact downscaled RGB buffer the detector consumes, raw so the
            // eval runner needs no image decoder and no dependencies
            fs.writeFileSync(
              path.join(FRAMES, `${id}.bin`),
              Buffer.from(s.raw, 'base64'),
            );

            labels.push({
              id,
              // corners normalised to [0,1] of the frame, clockwise from
              // top-left of the face; null means "no cube in this frame"
              corners: s.corners,
              tags: s.tags || '',
              detW: s.detW,
              detH: s.detH,
              videoW: s.videoW,
              videoH: s.videoH,
              // what the detector said at capture time, for drift tracking
              predicted: s.predicted || null,
              at: s.at,
            });
            fs.writeFileSync(LABELS, `${JSON.stringify(labels, null, 1)}\n`);

            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({
              ok: true, id, count: labels.length,
              positives: labels.filter((l) => l.corners).length,
              negatives: labels.filter((l) => !l.corners).length,
            }));
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        });
      });
    },
  };
}
