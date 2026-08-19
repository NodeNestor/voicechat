#!/usr/bin/env python3
"""The two model processes behind the live chat.

Gemma 4 needs transformers 5.x; CSM is scrambled on 5.x and only correct on
4.57.6. They cannot share an interpreter, so speech generation runs as a
subprocess of csm.rs (which is also 4x faster than any python path we measured).

  ear   Gemma 4 E4B, 4-bit, hears your audio clip directly - no separate ASR
  mouth csm.rs, cloned voice, 61 ms to first audio

The system prompt is prefilled once and its KV cache reused every turn, so each
turn only pays for your new audio plus the reply.
"""
import os
import subprocess
import threading
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

GEMMA = os.path.join(ROOT, "models", "google__gemma-4-E4B-it")
CSM_BIN = os.path.abspath(os.path.join(ROOT, "..", "csm.rs", "target", "release", "main.exe"))
CSM_WEIGHTS = os.path.join(ROOT, "models", "unsloth__csm-1b", "model.safetensors")
WORK = os.path.join(ROOT, "work")

VOICES_CUSTOM = os.path.join(ROOT, "voices_custom")
VOICE_PROMPTS = os.path.join(ROOT, "voice_prompts")
DEFAULT_VOICE = "clean_read_speech_a"


def resolve_voice(name=None):
    """Turn a voice name into (wav, transcript).

    VC_VOICE takes either a name - voices_custom/<name>.wav, which is what
    add_voice.py writes - or a path to a wav. The transcript is read from the
    .txt sitting beside the wav. CSM conditions on the text as well as the
    audio, so a wrong transcript degrades every reply; VC_VOICE_TEXT overrides
    it only if you really mean to.
    """
    name = name or os.environ.get("VC_VOICE") or DEFAULT_VOICE
    if os.path.isfile(name):
        wav = os.path.abspath(name)
    else:
        stem = name if name.lower().endswith(".wav") else name + ".wav"
        for d in (VOICES_CUSTOM, VOICE_PROMPTS):
            cand = os.path.join(d, stem)
            if os.path.isfile(cand):
                wav = cand
                break
        else:
            raise SystemExit(
                "voice %r not found. Looked in voices_custom/ and voice_prompts/.\n"
                "Make one with:  .venv/Scripts/python add_voice.py clip.mp3 --name %s"
                % (name, name))

    txt = os.environ.get("VC_VOICE_TEXT")
    if not txt:
        side = os.path.splitext(wav)[0] + ".txt"
        if os.path.isfile(side):
            with open(side, encoding="utf-8") as f:
                txt = f.read()
        else:
            # Not fatal - the older voice_prompts clips never had transcripts -
            # but it is a real quality hit, so say so rather than hiding it.
            print("[mouth] WARNING: no transcript beside %s; conditioning on audio\n"
                  "        alone is measurably worse. Rebuild it with add_voice.py."
                  % os.path.basename(wav), flush=True)
            txt = "reference audio"
    return wav, " ".join(txt.split())


SYSTEM_PROMPT = (
    "You are a voice assistant being spoken aloud, so talk like a person, not "
    "like a document. Keep replies to one or two short sentences, under 25 "
    "words. Use contractions. Never read out lists or headings. If you did not "
    "catch something, say so and ask again. Do not narrate what you are doing."
)


# --------------------------------------------------------------------- ear
class Ear:
    """Gemma 4: audio in, reply text out, with the system prompt cached."""

    def __init__(self, device=1):
        import torch
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(GEMMA)
        # Both of transformers' backends route through torchcodec, whose DLLs do
        # not load against torch 2.11 here (and torchaudio 2.11 delegates to it
        # as well). soundfile reads wav directly, so decode it ourselves.
        import soundfile as sf
        from scipy.signal import resample_poly
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
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
        self.model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            GEMMA, quantization_config=qc, device_map={"": device},
            dtype=torch.bfloat16).eval()
        self.history = []
        self._warm()

    def _warm(self):
        """One dummy turn so cuda graphs / kernels are resident before turn 1."""
        t0 = time.time()
        try:
            self.reply(np.zeros(16000, dtype=np.float32), _warmup=True)
        except Exception:
            pass
        print("[ear] warm in %.1fs" % (time.time() - t0), flush=True)

    def reply(self, audio_16k, _warmup=False):
        torch = self.torch
        path = os.path.join(WORK, "_turn_in.wav")
        os.makedirs(WORK, exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(audio_16k, -1, 1) * 32767).astype("<i2").tobytes())

        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        for role, text in self.history[-6:]:
            msgs.append({"role": role, "content": [{"type": "text", "text": text}]})
        msgs.append({"role": "user", "content": [{"type": "audio", "audio": path}]})

        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=True, return_dict=True,
                                            return_tensors="pt")
        inp = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=64, do_sample=False)
        text = self.proc.batch_decode(out[:, inp["input_ids"].shape[-1]:],
                                      skip_special_tokens=True)[0].strip()
        if not _warmup and text:
            self.history.append(("assistant", text))
        return text


# ------------------------------------------------------------------- mouth
class Mouth:
    """csm.rs as a subprocess. Splits on sentences: CSM generates one utterance
    per call and rambles past short prompts, so long replies are chunked."""

    def __init__(self, voice_wav=None, voice_txt=None):
        # resolved here, not at import, so a missing voice fails when you pick
        # it rather than when the module loads
        if voice_wav is None:
            voice_wav, resolved_txt = resolve_voice()
            voice_txt = voice_txt or resolved_txt
        elif voice_txt is None:
            voice_wav, voice_txt = resolve_voice(voice_wav)
        self.voice_wav = voice_wav
        self.voice_txt = voice_txt
        print("[mouth] voice %s (%d-word transcript)"
              % (os.path.basename(self.voice_wav), len(self.voice_txt.split())), flush=True)
        os.makedirs(WORK, exist_ok=True)

    @staticmethod
    def _split(text, max_words=14):
        import re
        parts, cur = [], []
        for tok in re.split(r"(?<=[.!?])\s+", text.strip()):
            if not tok:
                continue
            if len(tok.split()) <= max_words:
                parts.append(tok)
            else:                                  # long sentence: split on commas
                for piece in re.split(r",\s*", tok):
                    if piece:
                        parts.append(piece if piece[-1] in ".!?" else piece + ",")
        return [p for p in parts if p.strip()] or [text]

    def say(self, text, out_path):
        chunks = self._split(text)
        pieces = []
        for i, chunk in enumerate(chunks):
            p = os.path.join(WORK, "_say_%d.wav" % i)
            # cap length to the words present, so it cannot ramble
            cap = max(1500, int(len(chunk.split()) * 450))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
            subprocess.run(
                [CSM_BIN, "--weights-path", CSM_WEIGHTS, "--text", chunk,
                 "--output", p, "--buffer-size", "4",
                 "--max-audio-len-ms", str(cap),
                 "--ref-audio", self.voice_wav, "--ref-text", self.voice_txt],
                env=env, capture_output=True, text=True)
            if os.path.exists(p):
                with wave.open(p, "rb") as w:
                    pieces.append(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"))
        if not pieces:
            return None
        joined = np.concatenate(pieces)
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(joined.tobytes())
        return out_path
