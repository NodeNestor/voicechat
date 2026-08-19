# voicechat

A self-hosted voice assistant that runs in a browser tab. You speak into the
microphone, a local model receives the audio, and the reply is spoken back in a
voice cloned from a single reference clip.

All three models run on the local machine. There are no external API calls and
no audio leaves the host.

It was built to see how close a local setup gets to the interaction style of
Sesame's CSM demo. It is a hobby project.

## Components

Four services. Three hold models, one is the glue and the browser UI.

| service | what it is | port |
|---|---|---|
| `ear` | Gemma 4 E4B via llama.cpp — takes audio as input directly | 8781 |
| `mouth` | CSM-1B via [csm.rs](https://github.com/cartesia-one/csm.rs) — takes text incrementally | 8770 |
| `stt` | faster-whisper — produces a transcript | 8790 |
| `web` | the glue and the browser UI (stdlib Python, no framework) | 8800 |

## How a turn works

1. **The browser detects end of speech.** Microphone audio is captured
   continuously and gated in the page: it compares RMS against an adjustable
   threshold and ends the turn after a configurable period of silence. Nothing
   is uploaded while you are still speaking.

2. **The clip is posted.** The page encodes a wav and `POST`s it to `/turn`.
   Everything after this returns over a single SSE connection, so transcript,
   reply text, and audio all arrive on one channel.

3. **Whisper transcribes it.** This runs first, taking roughly 160 ms on a
   second GPU. The transcript is not the input to the reply — it is used for
   display and conversation history, and passed to the ear as an additional
   hint. If this service is unavailable the rest still works.

4. **The ear receives the audio.** The wav is sent to Gemma 4 base64-encoded as
   audio, along with the system prompt, a one-line time-and-place note, and the
   conversation history. Gemma 4 is multimodal over audio, so there is no
   transcribe-then-reason step; the transcript accompanies the audio rather than
   replacing it.

5. **The reply streams token by token.** Each fragment is published to the
   browser as a `delta` and simultaneously pushed onto a queue feeding the
   voice.

6. **Synthesis overlaps generation.** A single chunked HTTP connection to
   `csm.rs` stays open for the whole reply and text fragments are written into
   it as they arrive. CSM continues one generation rather than starting a new
   one per sentence, so its KV cache is not cleared mid-reply.

7. **Audio returns as it is produced,** republished over SSE as base64 PCM. The
   browser queues and plays chunks as they arrive.

## Latency

Throughput is not high — CSM generates at roughly 0.9–1.0x realtime on the
hardware this was developed on. Perceived latency is lower than that number
suggests because the stages overlap.

**Stages are not serialised.** A transcribe-then-generate-then-synthesise
pipeline has latency equal to the sum of its stages. Here transcription happens
during silence the browser was already waiting out, and synthesis runs while
generation is still producing tokens. Time to first audio is approximately the
ear's first few tokens plus CSM's first frame.

**The reply does not wait on ASR.** Because Gemma receives audio directly, a
wrong transcript degrades a hint rather than corrupting the input. In an
ASR-first pipeline the language model never sees the audio, so a misrecognised
word is unrecoverable.

**One connection to the voice rather than one per sentence.** An earlier version
issued a separate request per clause. It was slower, because each request
repeated model setup, and it produced audible discontinuities at each join
because generation restarted. Streaming into one generation avoids both.

**History is appended, not windowed.** A sliding window changes the prompt
prefix every turn and invalidates the KV cache behind it — measured at 71% more
prefill across 12 turns, increasing with conversation length. Appending keeps
per-turn prefill approximately flat.

**Speech starts on a phrase.** When no punctuation has arrived, four words are
enough to begin synthesising.

## Relationship to Sesame's demo

This is inference from the constraints, not knowledge of their implementation.
Their system is not open.

The general shape is what the problem requires: generation and synthesis have to
overlap, the speech model has to continue one generation rather than restart,
and letting the language model receive audio avoids an ASR bottleneck in the
reasoning path.

The differences are likely substantial and mostly in training rather than
architecture: fine-tuning both models for conversational speech, end-to-end
training instead of three separate models connected over HTTP, a speech model
trained to handle interruption and backchannelling, and prosody carried across
turns. This project is the plumbing, run locally, with off-the-shelf weights.

## Custom voices

CSM has no fixed speaker. Without a reference it samples a speaker identity from
its seed. Supplying a reference clip conditions it on that voice instead, which
changes the output substantially.

```bash
python add_voice.py myclip.mp3 --name alice
```

Accepts any format ffmpeg can decode. It will:

- resample to 24 kHz with a polyphase filter (linear interpolation aliases, and
  the model reproduces the aliasing)
- remove DC offset, normalise, and select a window that begins and ends in
  silence
- transcribe the trimmed clip, so the conditioning text matches the audio
- generate a test line

Then point the mouth at it:

```bash
export VC_VOICE=alice
```

### Reference clip characteristics

The reference is what the model imitates, so its defects appear in the output.

- **10–20 seconds.** Longer costs latency without a corresponding gain.
- **Clean audio.** Room reverb, background music, and compression artifacts are
  reproduced.
- **Connected speech**, not isolated words.
- **An accurate transcript.** Conditioning uses text and audio together.
  `add_voice.py` transcribes automatically; a supplied transcript should include
  disfluencies, as a tidied one measurably degrades the result.

Only clone voices you have the right to clone.

## Requirements

- **An NVIDIA GPU faster than an RTX 5060 Ti.** Capacity is not the constraint —
  16 GB is sufficient, with the ear at ~7 GB and the mouth at ~3.5 GB. Speed is
  the constraint; see [Performance](#performance).
- A second GPU is optional. It allows Whisper to run without competing with the
  ear; a few GB is enough.
- CUDA 12.x to build csm.rs (`cudarc` rejects 13.x).
- Python 3.11+, and Rust for csm.rs.

Model weights are downloaded separately and are not redistributed here.

## Setup

`csm.rs` is a separate AGPL-3.0 project. Clone and build it rather than
vendoring it, which keeps its license off this repository:

```bash
git clone https://github.com/cartesia-one/csm.rs
cd csm.rs && CUDA_COMPUTE_CAP=86 cargo build --release --features cuda --bin server
```

Set `CUDA_COMPUTE_CAP` for the target card: 80 for A100, 86 for Ampere consumer,
89 for Ada, 120 for Blackwell. A binary built for one will not run on another.

Start the four services:

```bash
# 1. ear
llama-server -m gemma-4-E4B-it-Q8_0.gguf \
             --mmproj mmproj-gemma-4-E4B-it-BF16.gguf \
             --host 127.0.0.1 --port 8781 -ngl 99 -np 1

# 2. mouth
./csm.rs/target/release/server \
    --weights-path csm-1b/model.safetensors \
    --host 127.0.0.1 --port 8770 --buffer-size 1 \
    --ref-audio voices_custom/alice.wav \
    --ref-text "$(cat voices_custom/alice.txt)"

# 3. stt
python stt_server.py

# 4. web
python live_server.py
```

Then open <http://127.0.0.1:8800>.

The mouth's reference audio and transcript are fixed for the lifetime of the
process, so changing voice requires restarting it. Pass the transcript from the
`.txt` that `add_voice.py` wrote next to the wav; a mismatched transcript is a
common cause of degraded output.

Environment variables:

| var | default | effect |
|---|---|---|
| `VC_PORT` | 8800 | web UI port |
| `VC_EAR` / `VC_MOUTH` / `VC_STT_URL` | localhost | point at remote services |
| `VC_BUF` | 1 | frames decoded per emitted chunk |
| `VC_SPEAK_WORDS` | 4 | words buffered before synthesis starts |
| `VC_LATLON` / `VC_PLACE` | unset | enables a time-and-weather line; unset disables it |

Services bind to loopback. To run them on a remote GPU host, forward the ports
over SSH (`ssh -N -L 8770:127.0.0.1:8770 user@host`) rather than exposing them.

## Performance

On an RTX 5060 Ti, CSM generates at roughly **0.9–1.0x realtime**. This sits on
the boundary: replies that generate above 1x play continuously, and replies that
fall below it cause the buffer to drain and the audio to break up mid-sentence.
The result is inconsistent rather than uniformly slow. A card with headroom
above 1x is required for consistent playback.

A larger datacentre GPU does not help. Measured on an A100 80GB, throughput is
the same. Generation is a sequential chain of small operations — one backbone
pass plus one decoder pass per codebook, 32 per 80 ms frame — so the cost is
per-operation dispatch rather than arithmetic. Memory bandwidth utilisation
measures ~6%, and an A100 clocks lower than a consumer card. Clock speed and
dispatch overhead determine throughput here, not width or bandwidth.

Removing that overhead would require CUDA graph capture. candle does not
currently support it: stream capture fails on its allocation path with
`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

## Licensing

The code in this repository is MIT (see [LICENSE](LICENSE)). The components it
runs are separately licensed and are not included here:

- **csm.rs** — AGPL-3.0. It runs as a separate process reached over HTTP.
  Exposing this as a public network service has implications under the AGPL's
  network clause.
- **CSM-1B** — Sesame's terms.
- **Gemma** — Google's Gemma license, which carries use restrictions.
- **faster-whisper / Whisper** — their respective licenses.
