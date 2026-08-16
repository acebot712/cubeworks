"""Cube-face acquisition service — MLX on Apple silicon.

Why a service at all: the classical detector's cold search costs 75-100ms
because it exhaustively combs the frame for every rotation and seam period.
Once the face is found, TRACKING it is 3.2ms and resolution-independent. So the
only expensive step is acquisition, and that is exactly the step a pretrained
model does well: one forward pass finds an object at any scale or angle.

This service therefore answers one question — "where is the cube face in this
frame?" — and returns a box. Everything downstream (exact corners, rectifying,
reading colours, validating the cube) stays in the browser, where it is
deterministic and testable.

Run:
    ./.venv-mlx/bin/python mlx-detect/server.py            # default model
    ./.venv-mlx/bin/python mlx-detect/server.py --port 8765

POST /detect  {"image": "data:image/jpeg;base64,..."}
  -> {"found": true, "box": [x0, y0, x1, y1], "norm": true, "ms": 812}
GET  /health  -> {"ok": true, "model": "...", "loaded": true}
"""
import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import segment

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
# Ask for the FACE, not the cube. A box around the whole 3D cube includes the
# side faces, so it is systematically larger and offset from the front face —
# measured box IoU 0.31-0.57 against the labelled face. The downstream search
# only needs a region, but a tighter region is a better one.
PROMPT = (
    "This image contains a Rubik's cube. Give the bounding box of ONLY the "
    "single face pointing most directly at the camera - the flat square grid of "
    "coloured squares facing the viewer. Do not include the side faces of the "
    "cube that are angled away, and do not include the hand. "
    "Respond with only JSON: {\"bbox_2d\": [x1, y1, x2, y2]}. "
    "If there is no Rubik's cube at all, respond {\"bbox_2d\": null}."
)

_state = {"model": None, "processor": None, "config": None, "name": DEFAULT_MODEL}


def load(name):
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config

    print(f"loading {name} ...", flush=True)
    t0 = time.time()
    model, processor = vlm_load(name)
    _state.update(model=model, processor=processor, config=load_config(name), name=name)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)


def parse_box(text, w, h):
    """Qwen-style grounding returns pixel coords in the RESIZED frame; models
    vary, so accept both pixel and 0-1000 normalised conventions and clamp."""
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    # heuristic: values well above the image size are the 0-1000 convention
    if max(v) > max(w, h) * 1.5:
        v = [v[0] / 1000 * w, v[1] / 1000 * h, v[2] / 1000 * w, v[3] / 1000 * h]
    x0, y0, x1, y1 = v
    x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0 / w, y0 / h, x1 / w, y1 / h]  # normalised, resolution-agnostic


def detect(image_bytes, prompt_text=None):
    from PIL import Image
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    prompt = apply_chat_template(_state["processor"], _state["config"], prompt_text or PROMPT, num_images=1)
    t0 = time.time()
    out = generate(
        _state["model"], _state["processor"], prompt, [img],
        max_tokens=96, temperature=0.0, verbose=False,
    )
    text = out if isinstance(out, str) else getattr(out, "text", str(out))
    box = parse_box(text, w, h)
    res = {
        "found": box is not None,
        "box": box,
        "norm": True,
        "ms": int((time.time() - t0) * 1000),
        "raw": text[:200],
    }
    # The VLM finds; SAM measures. A box is enough to prompt SAM, and its mask
    # gives the four corners the box never could.
    if box is not None and segment.available():
        t1 = time.time()
        try:
            fine = segment.refine(img, box)
        except Exception as e:
            fine = None
            res["segment_error"] = f"{type(e).__name__}: {e}"
        if fine:
            res["corners"] = fine["corners"]
            res["sam_score"] = fine["sam_score"]
            res["candidates"] = fine.get("candidates", [])
        res["segment_ms"] = int((time.time() - t1) * 1000)
    return res


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {
                "ok": True, "model": _state["name"],
                "loaded": _state["model"] is not None,
                "segment": segment.available(),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/detect"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(n))
            data = req["image"]
            prompt_override = req.get("prompt")
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            self._send(200, detect(base64.b64decode(data), prompt_override))
        except Exception as e:  # a bad frame must not take the service down
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-segment", action="store_true", help="skip SAM corner refinement")
    ap.add_argument("--sam", default=segment.MODEL, help="SAM checkpoint (vit-base is 6x smaller)")
    args = ap.parse_args()
    load(args.model)
    if args.no_segment:
        print("segmentation disabled", flush=True)
    elif segment.load(args.sam):
        print(f"SAM loaded ({args.sam}) — returning exact corners", flush=True)
    else:
        print(f"SAM unavailable, boxes only: {segment._state['error']}", flush=True)
    print(f"listening on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
