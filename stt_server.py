#!/usr/bin/env python3
"""faster-whisper behind a tiny HTTP endpoint.

The ear (Gemma) listens to your audio directly and does not need a transcript to
answer. This exists anyway because having the words is useful: they go into the
conversation history, they are shown in the browser, and they give the ear
something textual to anchor on when the audio is ambiguous.

It is deliberately small and fails open - if this process is down, the assistant
still works, it just has no transcript to display.

    POST /transcribe   raw 16-bit PCM wav in the body  ->  {"text": "..."}
    GET  /health                                       ->  {"ok": true}

Run:  python stt_server.py
Env:  VC_STT_PORT   (default 8790)
      VC_STT_MODEL  (default base.en; try small.en for better accuracy)
      VC_STT_DEVICE (default cuda; cpu works, slower)
"""
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("VC_STT_PORT", "8790"))
MODEL = os.environ.get("VC_STT_MODEL", "base.en")
DEVICE = os.environ.get("VC_STT_DEVICE", "cuda")
# int8_float16 keeps base.en well under a gigabyte, which is what lets it share
# a card with the ear without either of them having to move.
COMPUTE = os.environ.get("VC_STT_COMPUTE", "int8_float16" if DEVICE == "cuda" else "int8")


def log(msg):
    sys.stderr.write("[stt] %s\n" % msg)
    sys.stderr.flush()


log("loading faster-whisper %s on %s (%s)..." % (MODEL, DEVICE, COMPUTE))
try:
    from faster_whisper import WhisperModel
except ImportError:
    log("faster-whisper is not installed:  pip install faster-whisper")
    raise
_t0 = time.time()
model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
log("ready in %.1fs" % (time.time() - _t0))


def transcribe(wav_bytes):
    segments, _info = model.transcribe(
        io.BytesIO(wav_bytes),
        beam_size=1,              # greedy: this runs on the turn path
        vad_filter=True,          # drop the silence the browser sent along
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"ok": True, "model": MODEL})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/transcribe":
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        wav = self.rfile.read(n) if n else b""
        if not wav:
            self._json({"text": ""})
            return
        t0 = time.time()
        try:
            text = transcribe(wav)
        except Exception as e:
            # fail open: no transcript is better than a broken turn
            log("transcribe failed: %s" % str(e)[:120])
            self._json({"text": "", "error": str(e)[:200]})
            return
        log("%.0f ms  %r" % ((time.time() - t0) * 1000, text[:60]))
        self._json({"text": text})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    log("listening on http://127.0.0.1:%d" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("bye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
