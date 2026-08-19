#!/usr/bin/env python3
"""Add a custom voice from any audio clip.

    .venv/Scripts/python add_voice.py mysample.mp3 --name emma

That is the whole thing: one file in, a usable voice out. Give the transcript
yourself if you have it, otherwise Gemma 4 reads the clip back and writes it:

    .venv/Scripts/python add_voice.py clip.mp3 "what is said" --name emma
    .venv/Scripts/python add_voice.py clip.mp3 --transcript-file said.txt --name emma

Handles the whole recipe: decode via ffmpeg if needed, polyphase resample to
24 kHz (never linear - that aliases and the model copies the aliasing), downmix,
DC-remove, cut a window with silent edges, normalise, then generate a test line
so you can hear the result immediately.

The transcript matters. Conditioning uses text and audio together, so a wrong or
approximate transcript measurably degrades the clone. Two consequences worth
knowing:

  - When you supply the transcript, it must describe the audio that SURVIVES the
    trim, not the file you handed in. Pass --seconds large enough to keep the
    whole clip, or the text and audio silently stop matching. The script warns
    when it trims a clip whose transcript you supplied.
  - When you do not supply one, ASR runs on the TRIMMED clip for the same reason.

Run this with the VENV python: transcription needs transformers 5.x for Gemma.
Speech generation is a subprocess of csm.rs, so the 4.57.6 pin does not apply.
"""
import argparse
import os
import shutil
import subprocess
import wave

import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.abspath(__file__))
VOICES = os.path.join(ROOT, "voices_custom")
CSM_BIN = os.path.join(ROOT, "..", "csm.rs", "target", "release", "main.exe")
WEIGHTS = os.path.join(ROOT, "models", "unsloth__csm-1b", "model.safetensors")
GEMMA = os.path.join(ROOT, "models", "google__gemma-4-E4B-it")
SR = 24000


def decode_to_wav(src, dst):
    """Anything ffmpeg understands -> 16-bit mono wav at 24 kHz."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise SystemExit("ffmpeg not found on PATH; convert to 16-bit wav yourself")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", dst],
                   check=True)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if width != 2:
        raise SystemExit("need 16-bit input")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def write_wav(path, a, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())


def prep(a, sr, want_sec):
    """Returns (audio, trimmed) so the caller can warn about transcript drift."""
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
        print("  resampled %d -> %d Hz (anti-aliased)" % (sr, SR))
    a = a - a.mean()

    trimmed = False
    win = int(SR * 0.02)
    n = len(a) // win
    env = np.array([np.sqrt(np.mean(a[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    want = int(want_sec * SR / win)
    if want >= len(env):
        print("  keeping all %.1fs (no trim)" % (len(a) / SR))
    else:
        best, score_best = 0, -1e9
        for s in range(len(env) - want):
            e = s + want
            score = env[s:e].mean() - 2.0 * (env[s] + env[e - 1])
            if score > score_best:
                score_best, best = score, s
        a = a[best * win:(best + want) * win]
        trimmed = True
        print("  cut %.1fs starting at %.1fs (edges in silence)" % (len(a) / SR, best * win / SR))

    pk = np.abs(a).max()
    if pk > 0:
        a = a / pk * 0.95
    return a, trimmed


def _patch_audio_loader():
    """transformers decodes audio through torchcodec, whose DLLs do not load
    against torch 2.11 here - and torchaudio 2.11 delegates to it as well, so
    neither backend works. soundfile reads the file directly. Same bypass as
    voice_engine.Ear; keep the two in step."""
    import soundfile as sf
    from math import gcd

    def _load_audio_sf(audio, sampling_rate=16000, timeout=None, backend="auto"):
        if isinstance(audio, np.ndarray):
            return audio
        data, sr = sf.read(audio, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != sampling_rate:
            g = gcd(int(sr), int(sampling_rate))
            mono = resample_poly(mono, sampling_rate // g, sr // g).astype(np.float32)
        return mono

    import transformers.audio_utils as _au
    _au.load_audio = _load_audio_sf
    import transformers.processing_utils as _pu
    if hasattr(_pu, "load_audio"):
        _pu.load_audio = _load_audio_sf


def transcribe(wav_path, device=1):
    """Read the clip back with Gemma 4."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    if not os.path.isdir(GEMMA):
        raise SystemExit("gemma weights missing at %s - pass the transcript yourself" % GEMMA)

    _patch_audio_loader()
    print("loading gemma for transcription (4-bit)...")
    proc = AutoProcessor.from_pretrained(GEMMA)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        GEMMA, quantization_config=qc, device_map={"": device},
        dtype=torch.bfloat16).eval()

    msgs = [{"role": "user", "content": [
        {"type": "audio", "audio": wav_path},
        {"type": "text", "text": "Transcribe this speech word for word. "
                                 "Output only the transcript, no commentary."}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=True, return_dict=True, return_tensors="pt")
    inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    text = proc.batch_decode(out[:, inp["input_ids"].shape[-1]:],
                             skip_special_tokens=True)[0].strip()

    del model
    torch.cuda.empty_cache()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="any audio file: wav, mp3, m4a, ...")
    ap.add_argument("transcript", nargs="?", default=None,
                    help="exactly what is said in the clip (omit to transcribe with Gemma)")
    ap.add_argument("--transcript-file", default=None,
                    help="read the transcript from a file instead of the command line")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="length of window to keep; raise it to keep a whole clip")
    ap.add_argument("--device", type=int, default=1, help="cuda device")
    ap.add_argument("--say", default="Hey. So what are we working on today?",
                    help="test line to generate in the new voice")
    args = ap.parse_args()

    transcript = args.transcript
    if args.transcript_file:
        if transcript is not None:
            raise SystemExit("pass a transcript or --transcript-file, not both")
        with open(args.transcript_file, encoding="utf-8") as f:
            transcript = " ".join(f.read().split())
        print("transcript: %d words from %s"
              % (len(transcript.split()), os.path.basename(args.transcript_file)))

    os.makedirs(VOICES, exist_ok=True)
    src = args.audio
    if not os.path.exists(src):
        raise SystemExit("no such file: %s" % src)
    tmp = os.path.join(VOICES, "_tmp_%s.wav" % args.name)

    if not src.lower().endswith(".wav"):
        print("decoding %s with ffmpeg..." % os.path.basename(src))
        decode_to_wav(src, tmp)
        src = tmp

    a, sr = read_wav(src)
    print("source: %.2f s @ %d Hz" % (len(a) / sr, sr))
    a, trimmed = prep(a, sr, args.seconds)

    ref_wav = os.path.join(VOICES, "%s.wav" % args.name)
    ref_txt = os.path.join(VOICES, "%s.txt" % args.name)
    write_wav(ref_wav, a, SR)
    if os.path.exists(tmp):
        os.remove(tmp)

    if transcript is None:
        # ASR the trimmed clip: the text must describe the conditioned audio.
        transcript = transcribe(ref_wav, device=args.device)
        print("\n--- transcript ---\n%s\n" % transcript)
        if not transcript:
            raise SystemExit("transcription came back empty - pass the transcript yourself")
    elif trimmed:
        print("\n  WARNING: you supplied a transcript but the clip was trimmed to"
              " %.1fs.\n  The text now describes audio that is no longer there."
              " Re-run with --seconds %d to keep all of it."
              % (len(a) / SR, int(len(a) / SR) + 30))

    with open(ref_txt, "w", encoding="utf-8") as f:
        f.write(transcript)
    print("voice saved: %s" % ref_wav)
    print("            %s" % ref_txt)
    print("\nuse it:  set VC_VOICE=%s" % args.name)

    if not os.path.exists(CSM_BIN):
        print("\n(csm.rs not built - skipping the test render)")
        return
    out = os.path.join(VOICES, "%s_test.wav" % args.name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    print("\ngenerating a test line...")
    r = subprocess.run([CSM_BIN, "--weights-path", WEIGHTS, "--text", args.say,
                        "--output", out, "--buffer-size", "4",
                        "--ref-audio", ref_wav, "--ref-text", transcript],
                       env=env, capture_output=True, text=True)
    if os.path.exists(out):
        print("wrote %s  <- listen to this" % out)
    else:
        print("generation failed:\n%s" % (r.stderr or "")[-400:])


if __name__ == "__main__":
    main()
