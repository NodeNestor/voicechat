# voicechat

A local, self-hosted voice assistant you talk to in a browser tab. You speak, it
hears you, it answers out loud — in a voice you clone from a single audio clip.

This started as an attempt to get somewhere near the feel of
[Sesame's online CSM demo](https://www.sesame.com/) on hardware you own, with
nothing leaving the machine. It is a hobby project, not a product. It is also
genuinely fun to talk to.

Everything runs locally: the model that hears you, the model that answers, and
the model that speaks. No API keys, no accounts, no audio leaving the box.

## How it works

Four small services. Three hold models, one is the glue and the browser UI.

| service | what it is | port |
|---|---|---|
| `ear` | Gemma 4 E4B via llama.cpp — hears your audio *directly*, no separate ASR step | 8781 |
| `mouth` | CSM-1B via `csm.rs` — speaks, taking text incrementally | 8770 |
| `stt` | faster-whisper — a transcript for grounding and history | 8790 |
| `web` | the glue plus the browser UI (stdlib Python, no framework) | 8800 |

## How a turn works, end to end

1. **The browser listens.** Mic audio is captured continuously and gated
   locally: it watches RMS against a threshold you can drag, and calls the end
   of your turn after a hang time of silence. Nothing is sent while you are
   still talking.

2. **It posts the clip.** On end-of-speech the page encodes a wav and `POST`s it
   to `/turn`. Everything after this streams back over one SSE connection, so
   the browser gets transcript, text, and audio on the same channel.

3. **Whisper transcribes it — off the critical path.** This runs first and costs
   around 160 ms on a second GPU, which is roughly the silence the browser has
   already waited out, so it is close to free. Its output is *not* the input to
   the reply; it is there for display, for conversation history, and as a
   textual hint alongside the audio. If this service is down the assistant still
   works.

4. **The ear hears the audio itself.** The wav goes to Gemma 4 base64-encoded,
   as audio, together with the system prompt, a one-line time-and-place note,
   and the conversation history. Gemma is natively multimodal over audio, so
   there is no transcribe-then-reason step: it hears tone, hesitation, and
   half-finished words directly, and the transcript rides along as a hint rather
   than as the whole truth.

5. **The reply streams out token by token,** and each fragment does two things
   at once: it is published to the browser as a `delta` so the text appears as
   it is written, and it is pushed onto a queue feeding the voice.

6. **The mouth speaks while the ear is still writing.** A single chunked HTTP
   connection to `csm.rs` stays open for the whole reply, and text fragments are
   fed into it as they arrive. CSM continues *one* generation rather than
   starting a new one per sentence, so its KV cache is never cleared mid-reply.

7. **Audio comes back as it is produced** and is republished over SSE as base64
   PCM. The browser queues and plays chunks as they land. By the time the ear
   finishes its last word, most of the reply has already been spoken.

## Why it feels fast

It is worth being precise, because the honest throughput number below is *not*
fast: CSM generates at roughly 0.9x realtime. The system feels responsive
anyway, because the architecture hides latency rather than removing it.

**Nothing waits for the previous stage to finish.** The naive pipeline is
serial — transcribe, then think, then synthesize, then play — and its latency is
the sum of four stages. Here, transcription overlaps the silence you were
already leaving, and synthesis overlaps generation. Time to first audio is
roughly "ear's first few words" plus "CSM's first frame", not the sum of
everything.

**Skipping the ASR bottleneck in the reasoning path.** Because Gemma takes audio
directly, the reply does not wait on a transcript, and quality does not collapse
when the transcript is wrong. A mis-heard word in an ASR-first pipeline is
unrecoverable — the model never sees the audio. Here it is only a bad hint.

**One connection to the voice, not one per sentence.** An earlier version issued
a fresh request per clause. It was slower, because every request paid model
setup again, and it *sounded* worse: restarting generation reset prosody and put
an audible wall at every join. Streaming into a single generation removes both.

**A prompt prefix that stays stable.** History is appended, never windowed. A
sliding window drops messages off the front each turn, which changes the prefix
and throws away the KV cache behind it — measured at 71% more prefill over 12
turns, and getting worse. Appending keeps each turn's prefill roughly flat no
matter how long you talk.

**Speech starts on a phrase, not a sentence.** When no punctuation has arrived
yet, four words are enough to start speaking — long enough for the model to
shape a phrase, short enough that you are not waiting on a full clause.

## Is this how Sesame does it?

Honestly: this is a guess, and it should be read as one. Their demo is not open,
and none of this is based on inside knowledge.

But the shape falls out of the constraints. If you want a voice assistant that
responds in conversational time, you need generation and synthesis to overlap,
you need the speech model to continue one generation rather than restart, and
you gain a lot by letting the reasoning model hear audio instead of reading a
transcript. Those are not clever tricks so much as the small number of things
that work.

Where the real thing is presumably far ahead is everything this project does
not do: substantial fine-tuning of both models for conversational speech,
end-to-end training rather than three separate models bolted together with HTTP,
a speech model trained for interruption and backchannelling, and prosody that
carries across turns instead of resetting. This repository is the plumbing, run
locally, with off-the-shelf weights. The gap between it and a well-tuned system
is mostly training, not architecture.

## The voice is the whole thing

Out of the box CSM has no fixed speaker — it invents one. It sounds fine.

**With a good custom reference clip it sounds dramatically better**, and that is
the difference between a demo and something you actually want to talk to. This
is the step worth spending ten minutes on.

```bash
python add_voice.py myclip.mp3 --name alice
```

That is the whole process. Give it any audio file — wav, mp3, m4a — and it will:

- decode it, resample to 24 kHz with a proper polyphase filter (never linear
  interpolation; that aliases, and the model faithfully reproduces the aliasing)
- DC-remove, normalise, and cut a window that starts and ends in silence
- transcribe it, so the conditioning text matches the audio
- render a test line so you can hear the result immediately

Then point the mouth at it:

```bash
export VC_VOICE=alice
```

### Getting a good reference clip

The reference is what the model imitates, so its flaws become the voice's flaws.

- **10–20 seconds** is the sweet spot. Longer costs latency for little gain.
- **Clean audio only.** Room echo, background music, and compression artifacts
  all get cloned faithfully.
- **Natural, connected speech.** Reading a list of words gives a voice that
  sounds like it is reading a list of words.
- **An accurate transcript matters.** Conditioning uses text *and* audio
  together. `add_voice.py` transcribes for you; if you pass your own, include
  the ums and false starts — a tidied-up transcript measurably degrades the
  clone.

Only clone voices you have the right to clone.

## Requirements

- NVIDIA GPU. Comfortable on 16 GB; the ear is ~7 GB and the mouth ~3.5 GB.
- CUDA 12.x for building csm.rs (`cudarc` does not accept 13.x).
- Python 3.11+, and Rust for csm.rs.

Models are downloaded separately and are **not** redistributed here — see
Licensing.

## Setup

`csm.rs` is a separate AGPL-3.0 project (a Rust/candle implementation of
Sesame's CSM). Clone and build it yourself rather than vendoring it — that keeps
its license off this repository:

```bash
git clone <csm.rs repository>
cd csm.rs && CUDA_COMPUTE_CAP=86 cargo build --release --features cuda --bin server
```

Set `CUDA_COMPUTE_CAP` for your card: 80 for A100, 86 for Ampere consumer,
89 for Ada (4090/4060), 120 for Blackwell (50-series). A binary built for one
will not run on another.

Then start the four services, each in its own terminal:

```bash
# 1. ear - Gemma 4 E4B, hears audio directly
llama-server -m gemma-4-E4B-it-Q8_0.gguf \n             --mmproj mmproj-gemma-4-E4B-it-BF16.gguf \n             --host 127.0.0.1 --port 8781 -ngl 99 -np 1

# 2. mouth - CSM, with your cloned voice
./csm.rs/target/release/server \n    --weights-path csm-1b/model.safetensors \n    --host 127.0.0.1 --port 8770 --buffer-size 1 \n    --ref-audio voices_custom/alice.wav \n    --ref-text "$(cat voices_custom/alice.txt)"

# 3. stt - transcripts for history and display
python stt_server.py

# 4. web - the glue and the UI
python live_server.py
```

Open <http://127.0.0.1:8800> and click **Listen**.

A note on the mouth: the reference audio and its transcript are passed at
startup and fixed for the life of the process, so restarting it is how you
change voice. Pass the transcript from the `.txt` that `add_voice.py` wrote
next to the wav rather than typing one - a mismatched transcript is the most
common cause of a clone sounding worse than it should.

Useful environment variables:

| var | default | what |
|---|---|---|
| `VC_PORT` | 8800 | web UI port |
| `VC_EAR` / `VC_MOUTH` / `VC_STT_URL` | localhost | point at remote services |
| `VC_BUF` | 1 | frames decoded per chunk; 1 is smoothest |
| `VC_SPEAK_WORDS` | 4 | words buffered before speech starts |
| `VC_LATLON` / `VC_PLACE` | unset | enables a time-and-weather line; unset = off |

The services bind to loopback. To run them on a remote GPU box, forward the
ports over SSH (`ssh -N -L 8770:127.0.0.1:8770 you@box`) rather than exposing
them.

## Performance, honestly

On an RTX 5060 Ti the mouth generates at roughly **0.9× realtime** — very
slightly slower than playback. Short replies are fine; long ones can drain the
buffer.

That number is not GPU-bound, and this is the interesting part. Measured on an
A100 80GB, it is *the same* — because generation is one strictly sequential
chain of small operations (a backbone pass plus one decoder pass per codebook,
32 of them, per 80 ms frame), and the cost is per-operation dispatch rather than
arithmetic. Memory bandwidth utilisation sits at ~6%. A bigger GPU does not
help; the fix would be CUDA graph capture, which candle does not currently
support (stream capture fails on its allocation path).

So: this is what the architecture costs today. It is good enough to hold a
conversation, and the system prompt keeps replies short partly for this reason.

## Licensing

The code in this repository is MIT (see [LICENSE](LICENSE)).

The things it runs are not, and are **not included here**:

- **csm.rs is AGPL-3.0.** It runs as a separate process reached over HTTP. If
  you expose this as a public network service, read the AGPL's network clause.
- **CSM-1B** weights come from Sesame under their own terms.
- **Gemma** is under Google's Gemma license, which carries use restrictions.
- **faster-whisper / Whisper** under their respective licenses.

Check each before doing anything beyond running it at home.

## Status

A weekend-shaped project that works. Rough edges are load-bearing, not
decorative.
