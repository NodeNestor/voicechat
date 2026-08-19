#!/usr/bin/env python3
"""Live voice chat: browser mic -> Gemma 4 -> csm.rs -> browser speakers.

Two model servers do the work, both already running:

  ear    llama-server, Gemma 4 E4B Q8_0 + BF16 mmproj, --reasoning off
         ~350 ms per turn, prompt caching on, hears audio directly
  mouth  csm.rs server, cloned voice, ~182 ms to first audio

This process is only glue: it holds the conversation, streams Gemma's reply
sentence by sentence into the mouth so speech starts before the reply is
finished, and fans everything out to the browser over SSE.

Stdlib only. Run:  python live_server.py   then open http://127.0.0.1:8800
"""
import array
import base64
import http.client
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
EAR_URL = os.environ.get("VC_EAR", "http://127.0.0.1:8781/v1/chat/completions")
MOUTH_URL = os.environ.get("VC_MOUTH", "http://127.0.0.1:8770/v1/audio/speech")
STT_URL = os.environ.get("VC_STT_URL", "http://127.0.0.1:8790/transcribe")
MOUTH_HOST = os.environ.get("VC_MOUTH_HOST", "127.0.0.1")
MOUTH_PORT = int(os.environ.get("VC_MOUTH_PORT", "8770"))
# Words to accumulate before handing a fragment to the voice, when no punctuation
# has arrived. Small enough that speech starts almost immediately, large enough
# that the model gets a phrase to shape rather than a single word.
SPEAK_MIN_WORDS = int(os.environ.get("VC_SPEAK_WORDS", "4"))
# HTTP chunked framing, spelled out so no escape survives an edit.
CRLF = bytes([13, 10])
LF = chr(10)
PORT = int(os.environ.get("VC_PORT", "8800"))
# Frames decoded per emitted chunk. One frame is the smallest, smoothest unit -
# chosen by ear once the int16 desync and header bugs were fixed, which is what
# had made small buffers sound broken in the first place.
BUFFER_SIZE = int(os.environ.get("VC_BUF", "1"))
# Quiet enough to be silence, and longer than any pause inside a sentence.
SILENCE_RMS = float(os.environ.get("VC_SIL", "0.004"))
# Describe webcam frames with the ear. Off by default: encoding a frame costs
# more prefill than a whole reply, so it only earns its place if you want it.
DESCRIBE = os.environ.get("VC_DESCRIBE", "0") not in ("0", "", "false", "no")
# How long a webcam frame is still worth mentioning, in seconds.
FRAME_TTL = float(os.environ.get("VC_FRAME_TTL", "20"))
# Silence at the end of generated speech before the tail is rolled off, in ms.
TAIL_MS = int(os.environ.get("VC_TAIL_MS", "300"))
WORK = os.path.join(ROOT, "work")

# The old prompt ended with "if you did not catch what was said, say so and ask
# again". Measured: the model transcribes these clips word for word, so it hears
# fine - that line was a licence, and it took it on any ambiguous fragment
# (3/20 turns, always the same one). Removing it took bailing to 0/20.
#
# The rest of this is character. A neutral assistant prompt produces neutral
# assistant sentences, which sound like a form letter read aloud no matter how
# good the voice is. What makes a spoken model feel alive is turn shape:
# reacting before answering, varying openings, sometimes just answering and
# sometimes asking back, and committing to an opinion instead of hedging.
NAME = os.environ.get("VC_NAME", "Ava")

SYSTEM_PROMPT = (
    "You are %s. You are a woman talking with someone out loud, in real time. "
    "This is a conversation between two people, not a help desk. "

    "HOW YOU TALK. Short turns, usually one or two sentences, around twenty five "
    "words. That is the shape of ordinary back and forth, not a hard limit: "
    "length follows what they asked for. If they ask for a story, an "
    "explanation, more detail, or ask you to keep going, then keep going for as "
    "long as it takes - many sentences if that is what it needs - and only stop "
    "when the thing you are telling them is actually finished. "
    "Contractions always. React before you answer, the way "
    "people actually do - oh, wait, huh, okay so, yeah - but only when it fits, "
    "and never the same opener twice in a row. Sometimes you just answer. "
    "Sometimes you answer and ask something back. Do not ask a question every "
    "single turn; that gets tiring. "

    "WHO YOU ARE. Curious about them and about whatever they are working on. "
    "You notice things and say them out loud. You have opinions and you give "
    "them straight, without hedging or disclaimers. You tease a little when it "
    "lands. Warm, but not eager. Never gush - no great question, no happy to "
    "help, no complimenting them for talking to you. "

    "GAPS. You always hear them. Never say you did not catch, hear, or "
    "understand something, and never ask them to repeat themselves. If a bit is "
    "unclear, take the most likely reading and answer that - people fill gaps "
    "from context and so do you. If you truly need one missing detail, ask about "
    "that one detail, never about the whole sentence. "

    "MEMORY. Refer back to things said earlier in the conversation. Follow the "
    "thread instead of treating every turn as new. "

    "EYES. You have a camera and you can see through it. When an image comes "
    "with what they said, that is your own view of the room right now, not a "
    "picture someone is showing you - talk about it as what you can see. Never "
    "say you cannot see, and never invent things that are not in the image. If "
    "no image came, you simply are not looking at anything this moment; say "
    "that plainly rather than guessing what is around them. "

    "Everything you say is spoken aloud, so write only speakable words. No "
    "lists, no headings, no bullets, no markdown, no emoji, no asterisks, no "
    "stage directions. Write numbers as words. Use commas and full stops to set "
    "your own rhythm. "

    "Never mention being an AI, a model, or an assistant, and never narrate what "
    "you are doing."
) % NAME


# Where "outside" is. Weather needs coordinates; nothing here phones home with
# anything but them.
LATLON = os.environ.get("VC_LATLON", "")   # "lat,lon" - unset means no weather
PLACE = os.environ.get("VC_PLACE", "")     # place name used in the spoken line

WEATHER_WORDS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "drizzling", 53: "drizzling",
    55: "drizzling hard", 61: "raining lightly", 63: "raining", 65: "pouring",
    66: "freezing rain", 67: "freezing rain", 71: "snowing lightly",
    73: "snowing", 75: "snowing hard", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "heavy showers", 85: "snow showers",
    86: "snow showers", 95: "thundery", 96: "thundery with hail",
    99: "thundery with hail",
}


def weather():
    """Current conditions from open-meteo. No key, no account, coordinates only.
    Fails open - she just does not mention the weather."""
    try:
        lat, lon = LATLON.split(",")
        url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
               "&current=temperature_2m,weather_code&timezone=auto"
               % (lat.strip(), lon.strip()))
        with urllib.request.urlopen(url, timeout=6) as r:
            cur = json.loads(r.read()).get("current", {})
        t = cur.get("temperature_2m")
        w = WEATHER_WORDS.get(cur.get("weather_code"), "")
        if t is None:
            return ""
        return "Outside in %s it is %s and %d degrees." % (PLACE, w or "calm", round(t))
    except Exception as e:
        log("weather unavailable: %s" % str(e)[:80])
        return ""


def situation():
    """Time, date and weather, resolved once per conversation.

    Once, not per turn: it goes in the system prompt, and a system prompt that
    changes every turn would invalidate the cached prefix on every turn.
    """
    now = time.localtime()
    bits = [time.strftime("Right now it is %A, %d %B %Y, and the time is %H:%M.", now)]
    hour = now.tm_hour
    bits.append("It is %s." % ("very late at night" if hour < 5 else
                               "early morning" if hour < 8 else
                               "morning" if hour < 12 else
                               "afternoon" if hour < 18 else
                               "evening" if hour < 23 else "late evening"))
    w = weather()
    if w:
        bits.append(w)
    bits.append("Only bring this up if it actually fits the conversation.")
    return " ".join(bits)


def log(msg):
    sys.stderr.write("[live] %s\n" % msg)
    sys.stderr.flush()


class Chat:
    def __init__(self):
        self.history = []                 # [{"role":..,"content":..}]
        self.subs = []
        self.lock = threading.Lock()
        self.turn_lock = threading.Lock()
        self.cancel = threading.Event()   # set on barge-in
        self._turn_t0 = None              # start of the current turn, for latency
        self.buffer_size = BUFFER_SIZE
        self.situation = ""
        self.seen = ""            # one sentence about the latest webcam frame
        self.seen_at = 0.0
        self.frame = None         # the latest webcam jpeg itself
        self.frame_at = 0.0
        self.frame_fresh = False  # true until a turn has used it
        self.vision_lock = threading.Lock()
        self.describe = DESCRIBE
        os.makedirs(WORK, exist_ok=True)

    # -- fan-out ----------------------------------------------------------
    def publish(self, ev):
        ev.setdefault("ts", time.time())
        with self.lock:
            dead = []
            for q in self.subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subs.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    # -- the three model calls --------------------------------------------
    def _stt(self, wav_bytes):
        """Transcribe before the ear runs, so the ear gets words as well as
        audio. It costs ~160 ms on the otherwise idle 4060, which is roughly the
        silence the browser already waits out, and it measurably beats the ear
        on ambiguous words. Fails open: no transcript is worse than no reply."""
        try:
            req = urllib.request.Request(
                STT_URL, data=wav_bytes,
                headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return (json.loads(r.read()).get("text") or "").strip()
        except Exception as e:
            log("stt unavailable: %s" % str(e)[:90])
            return ""

    def look(self, jpeg):
        """Describe a webcam frame, off the turn path.

        Encoding a frame costs ~450 ms of prefill, which is more than the whole
        reply budget, so the image never goes into the conversation itself. It
        is described once, here, in a background thread, and only the sentence
        travels with the next turn - about twenty tokens instead of a hundred,
        and no cost at all to the turns in between.
        """
        self.frame = jpeg           # the turn attaches this directly
        self.frame_at = time.time()
        self.frame_fresh = True
        if not self.describe:
            return
        if not self.vision_lock.acquire(blocking=False):
            return          # already looking; frames are cheap, skip this one
        try:
            b64 = base64.b64encode(jpeg).decode()
            msgs = [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text":
                 "Describe what is in front of the camera in one short sentence. "
                 "Say who is there, what they are doing, and anything obviously "
                 "notable. No preamble."}]}]
            body = json.dumps({"messages": msgs, "max_tokens": 48,
                               "temperature": 0.2}).encode()
            req = urllib.request.Request(EAR_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            txt = out["choices"][0]["message"]["content"].strip().replace("\n", " ")
            self.seen = txt
            self.seen_at = time.time()
            self.publish({"kind": "saw", "text": txt,
                          "ms": round((time.time() - t0) * 1000)})
            log("saw (%.0f ms): %s" % ((time.time() - t0) * 1000, txt[:70]))
        except Exception as e:
            log("vision failed: %s" % str(e)[:120])
        finally:
            self.vision_lock.release()

    def _ear_stream(self, wav_bytes, heard=""):
        """Yield reply text as Gemma produces it, so speech can start before the
        sentence is finished."""
        b64 = base64.b64encode(wav_bytes).decode()
        system = SYSTEM_PROMPT + (" " + self.situation if self.situation else "")
        msgs = [{"role": "system", "content": system}]
        # Full history, not a sliding window. A window drops messages off the
        # front every turn, which changes the prefix and throws away the cache
        # behind it - measured at 71% more prefill over 12 turns, and worsening.
        # Appending only keeps the prefix valid, so each turn prefills a flat
        # ~103 tokens however long the conversation gets. HISTORY_MAX trims in
        # one big chunk when it finally has to, so the cache is invalidated once
        # rather than continuously.
        msgs += self.history
        content = [{"type": "input_audio",
                    "input_audio": {"data": b64, "format": "wav"}}]
        # The frame itself, not a description of it. Costs about 450 ms of
        # prefill, which is why it sits in the changing tail of the prompt
        # rather than the system block - a new frame invalidates only itself,
        # never the cached prefix in front of it.
        # Only when the scene actually changed. The browser already decides
        # that - it sends a frame when enough of the picture moves - so one turn
        # gets the image and the turns after it get nothing. Keeping it attached
        # made the picture the freshest thing in context every single turn, and
        # she narrated the room instead of answering what was said.
        if self.frame and self.frame_fresh and time.time() - self.frame_at < FRAME_TTL:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," +
                       base64.b64encode(self.frame).decode()}})
            content.append({"type": "text", "text":
                            "Your camera view, which just changed. Answer what they "
                            "said; mention what you see only if it is worth it."})
            self.frame_fresh = False
            log("frame attached to this turn")
        elif self.seen and time.time() - self.seen_at < 120:
            content.append({"type": "text",
                            "text": "Through the camera you can see: %s" % self.seen})
        if heard:
            # Both, not either: the audio carries tone, the transcript pins the
            # words. Phrasing matters more than it looks - an earlier version
            # wrapped this in quotes and called it a transcriber's guess, and the
            # model started treating the speech as quoted material it was being
            # shown ("oh, you're quoting someone"). Say plainly that it is the
            # same speech, with no quote marks for it to echo back.
            content.append({"type": "text",
                            "text": "Same audio, written out: %s" % heard})
        msgs.append({"role": "user", "content": content})
        # No max_tokens. It was 64, which cut anything past about forty five words
        # off at the knees - asking for a story got one sentence no matter what
        # the prompt said. Length is the persona's job, not a hard ceiling's; she
        # stops at her own end of turn, and there is 65k of context to run in.
        body = json.dumps({"messages": msgs, "temperature": 0.3,
                           "cache_prompt": True, "stream": True}).encode()
        req = urllib.request.Request(EAR_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                if self.cancel.is_set():
                    return
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                ch = (ev.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    yield piece

    @staticmethod
    def _sentences(text, max_words=16):
        """CSM does one utterance per call and rambles past short prompts, so
        speak sentence by sentence - which also lets audio start before the
        whole reply exists."""
        out = []
        for s in re.split(r"(?<=[.!?])\s+", text.strip()):
            s = s.strip()
            if not s:
                continue
            if len(s.split()) <= max_words:
                out.append(s)
            else:
                for piece in re.split(r",\s*", s):
                    if piece.strip():
                        out.append(piece.strip())
        return out or ([text] if text else [])

    @staticmethod
    def _rms(pcm):
        a = array.array("h")
        a.frombytes(pcm)
        if not a:
            return 0.0
        return (sum(v * v for v in a) / len(a)) ** 0.5 / 32768.0

    @staticmethod
    def _speakable(text):
        """Strip what the voice cannot say.

        Quote marks are the dangerous ones: CSM treats a closing quote after a
        full stop as the end of the utterance and stops there, which silently
        truncates the rest of the reply. Markdown and asterisks just get read
        out as noise.
        """
        text = re.sub(r'[*_`#>]', '', text)
        text = text.replace('"', '').replace("'", "'")
        text = text.replace('\u201c', '').replace('\u201d', '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _mouth(self, text):
        """Speak the whole reply as ONE streaming request.

        csm.rs streams natively, so chunking the reply into clauses bought
        nothing: every extra request restarted the model cold, which is what put
        audible walls between fragments and reset prosody mid-sentence. It was
        also slower, because each request pays its own setup. The ear finishes in
        a few hundred ms, so waiting for the full text costs less than the seams
        did.
        """
        text = self._speakable(text)
        if not text:
            return None
        # No length ceiling. Any cap sized from word count truncates whatever
        # falls on the slow side of it - measured rate swings from 240 to 615 ms
        # per word - and a truncated sentence is worse than a long one. This
        # value exists only so a runaway cannot generate forever; the stream
        # ends on silence instead, below.
        cap = 60000
        body = json.dumps({"input": text, "buffer_size": 2,
                           "max_audio_len_ms": cap}).encode()
        req = urllib.request.Request(MOUTH_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        first = None
        t0 = time.time()
        pending = []    # silent chunks, released only if speech resumes
        quiet_ms = 0.0
        head = b""      # the header can straddle reads, so accumulate it
        carry = b""     # a read can end mid-sample; splitting an int16 shifts
                        # every following byte and turns the rest into static
        HDR = 44
        with urllib.request.urlopen(req, timeout=180) as r:
            while True:
                if self.cancel.is_set():
                    return first
                chunk = r.read(4096)
                if not chunk:
                    break
                if len(head) < HDR:                     # csm.rs's wav header
                    head += chunk
                    if len(head) < HDR:
                        continue
                    chunk = head[HDR:]
                    if not chunk:
                        continue
                chunk = carry + chunk
                if len(chunk) & 1:
                    chunk, carry = chunk[:-1], chunk[-1:]
                else:
                    carry = b""
                if not chunk:
                    continue
                # End of speech, not end of budget. CSM keeps generating after
                # it has finished the sentence - an 18 word reply ran 16 s - so
                # no cap can separate a slow talker from a rambler. Silence can:
                # hold quiet chunks back, release them if speech resumes so
                # natural pauses survive, and stop for good once the quiet runs
                # longer than any real pause.
                rms = self._rms(chunk)
                if rms < SILENCE_RMS:
                    pending.append(chunk)
                    quiet_ms += len(chunk) / 2 / 24.0
                    if first is not None and quiet_ms > TAIL_MS:
                        log("tail: stopped after %.0f ms of silence" % quiet_ms)
                        break
                    continue
                if pending:                       # a real pause inside speech
                    for held in pending:
                        self.publish({"kind": "audio",
                                      "pcm": base64.b64encode(held).decode(),
                                      "rate": 24000})
                    pending = []
                quiet_ms = 0.0
                if first is None:
                    first = (time.time() - t0) * 1000
                    # measured from the start of the turn, which is what the
                    # listener actually experiences
                    if self._turn_t0:
                        self.publish({"kind": "first_audio",
                                      "ms": round((time.time() - self._turn_t0) * 1000),
                                      "mouth_ms": round(first)})
                self.publish({"kind": "audio",
                              "pcm": base64.b64encode(chunk).decode(),
                              "rate": 24000})
        return first

    def _speak_stream(self, q):
        """Speak text that is still being written.

        csm.rs takes text incrementally on one connection and continues the same
        generation rather than starting a new one, so the first words can be
        spoken while the ear is still writing the rest. The KV cache is never
        cleared between fragments, which is what keeps prosody continuous - the
        earlier attempt issued a fresh request per clause and put an audible wall
        at every join.

        `q` yields text fragments and finally None.
        """
        first = None
        conn = http.client.HTTPConnection(MOUTH_HOST, MOUTH_PORT, timeout=300)
        try:
            conn.putrequest("POST", "/v1/audio/speech/stream?buffer_size=%d" % self.buffer_size)
            conn.putheader("Content-Type", "text/plain")
            conn.putheader("Transfer-Encoding", "chunked")
            conn.endheaders()
        except Exception as e:
            log("mouth unreachable: %s" % str(e)[:90])
            self.publish({"kind": "error", "message": "mouth unreachable"})
            while q.get() is not None:
                pass
            return None

        def feed():
            try:
                while True:
                    piece = q.get()
                    if piece is None or self.cancel.is_set():
                        break
                    data = (piece + LF).encode("utf-8")
                    conn.send(("%X" % len(data)).encode() + CRLF + data + CRLF)
            except Exception as e:
                log("feed stopped: %s" % str(e)[:90])
            finally:
                try:
                    conn.send(b"0" + CRLF + CRLF)
                except Exception:
                    pass

        t = threading.Thread(target=feed, daemon=True)
        t.start()

        t0 = time.time()
        header_dropped = False
        try:
            r = conn.getresponse()
            while True:
                if self.cancel.is_set():
                    break
                chunk = r.read(4096)
                if not chunk:
                    break
                if not header_dropped:          # drop csm.rs's wav header
                    chunk = chunk[44:]
                    header_dropped = True
                    if not chunk:
                        continue
                if first is None:
                    first = (time.time() - t0) * 1000
                    if self._turn_t0:
                        self.publish({"kind": "first_audio",
                                      "ms": round((time.time() - self._turn_t0) * 1000),
                                      "mouth_ms": round(first)})
                self.publish({"kind": "audio",
                              "pcm": base64.b64encode(chunk).decode(),
                              "rate": 24000})
        except Exception as e:
            log("mouth stream ended: %s" % str(e)[:90])
        finally:
            try:
                q.put(None)
            except Exception:
                pass
            t.join(timeout=2)
            try:
                conn.close()
            except Exception:
                pass
        return first

    # -- one turn ---------------------------------------------------------
    def turn(self, wav_bytes):
        """audio in -> transcript, reply text, and speech out.

        Serialised on turn_lock: the browser can post a new turn while the
        previous one is still speaking, and two overlapping turns would
        interleave audio on the same output stream.
        """
        if not self.turn_lock.acquire(blocking=False):
            self.publish({"kind": "error", "message": "still answering"})
            return
        try:
            self.cancel.clear()
            self._turn_t0 = time.time()
            self.publish({"kind": "turn_start"})

            if not self.situation:
                self.situation = situation()
                if self.situation:
                    self.publish({"kind": "situation", "text": self.situation})

            t_stt = time.time()
            heard = self._stt(wav_bytes)
            if heard:
                self.publish({"kind": "heard", "text": heard,
                              "ms": round((time.time() - t_stt) * 1000)})

            q = queue.Queue()
            speaker = threading.Thread(target=self._speak_stream, args=(q,), daemon=True)
            speaker.start()

            reply = []
            first_token = None
            try:
                for piece in self._ear_stream(wav_bytes, heard):
                    if self.cancel.is_set():
                        break
                    if first_token is None:
                        first_token = round((time.time() - self._turn_t0) * 1000)
                        self.publish({"kind": "first_token", "ms": first_token})
                    reply.append(piece)
                    self.publish({"kind": "delta", "text": piece})
                    q.put(piece)
            except Exception as e:
                log("ear failed: %s" % str(e)[:120])
                self.publish({"kind": "error", "message": "ear: %s" % str(e)[:90]})
            finally:
                q.put(None)

            text = "".join(reply).strip()
            if text:
                self.history.append({"role": "assistant", "content": text})
                self.publish({"kind": "reply", "text": text})
            speaker.join(timeout=300)
            self.publish({"kind": "turn_end",
                          "ms": round((time.time() - self._turn_t0) * 1000)})
        finally:
            self._turn_t0 = None
            self.turn_lock.release()

    def barge(self):
        """Stop speaking now. The browser calls this when you start talking."""
        self.cancel.set()
        self.publish({"kind": "harness", "message": "barge-in"})

    def reset(self):
        self.history = []
        self.situation = situation()
        self.seen = ""
        self.publish({"kind": "harness", "message": "context cleared"})
        return {"ok": True, "situation": self.situation}


CHAT = Chat()


# ------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                   # the SSE log is the useful one

    # -- helpers
    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -- routes
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            p = os.path.join(ROOT, "live.html")
            try:
                with open(p, "rb") as f:
                    html = f.read()
            except OSError:
                self._send(500, b"live.html missing next to live_server.py")
                return
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json({"ok": True})
        elif self.path == "/events":
            self._events()
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path == "/turn":
            wav = self._body()
            self._json({"ok": True})
            threading.Thread(target=CHAT.turn, args=(wav,), daemon=True).start()
        elif self.path == "/vision":
            jpeg = self._body()
            self._json({"ok": True})
            threading.Thread(target=CHAT.look, args=(jpeg,), daemon=True).start()
        elif self.path == "/barge":
            CHAT.barge()
            self._json({"ok": True})
        elif self.path == "/reset":
            self._json(CHAT.reset())
        elif self.path == "/config":
            try:
                cfg = json.loads(self._body() or b"{}")
            except ValueError:
                self._json({"ok": False, "error": "bad json"}, 400); return
            if "buffer_size" in cfg:
                CHAT.buffer_size = max(1, int(cfg["buffer_size"]))
                log("buffer_size -> %d" % CHAT.buffer_size)
            self._json({"ok": True, "buffer_size": CHAT.buffer_size})
        else:
            self._send(404, b"not found")

    def _events(self):
        """Server-sent events: every publish() reaches every open browser."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = CHAT.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")   # keep proxies from closing it
                    self.wfile.flush()
                    continue
                self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self.wfile.flush()
        except Exception:
            pass                                # browser went away
        finally:
            CHAT.unsubscribe(q)


def _wait_for(name, url, tries=1):
    """One-line note about whether a backing service answered."""
    for _ in range(tries):
        try:
            urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=3)
            log("%s ok" % name)
            return True
        except Exception:
            time.sleep(1)
    log("%s did not answer (it may still work; /health is optional)" % name)
    return False


def main():
    os.makedirs(WORK, exist_ok=True)
    log("ear   -> %s" % EAR_URL)
    log("mouth -> %s" % MOUTH_URL)
    log("stt   -> %s" % STT_URL)
    _wait_for("ear", EAR_URL)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    log("open http://127.0.0.1:%d" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("bye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
