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

The ear takes your recorded audio as input rather than a transcript, which is
why it handles mumbling and half-words better than a pipeline that transcribes
first and reasons second. The transcript is still produced alongside, because
having the words is useful for history and grounding.

Text streams into the mouth clause by clause while the ear is still writing, so
speech starts before the reply is finished.

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
