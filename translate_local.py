#!/usr/bin/env python3
"""
translate_local.py - Urdu/Hindi/Arabic audio or video -> bilingual SRT (original + English)

LOCAL FILES ONLY. No yt-dlp, no network access needed after the model is downloaded.

Usage:
    python translate_local.py interview.mp4
    python translate_local.py a.mp3 b.mkv c.wav --language ur --model large-v3
    python translate_local.py --folder ./recordings --output-dir ./subs
    python translate_local.py                      # interactive file picker

Install:
    pip install faster-whisper
    (NVIDIA GPU) also needs CUDA 12 + cuDNN 9 runtime libraries, e.g.
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   (see the faster-whisper README)
    No separate ffmpeg is needed: faster-whisper decodes media through PyAV.
"""

import argparse
import os
import sys
import time

import av

# PyAV compatibility fix: newer PyAV removed metadata_errors parameter expected by faster-whisper
_orig_av_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _safe_av_open

from faster_whisper import WhisperModel, decode_audio

SAMPLE_RATE = 16000
MEDIA_EXTS = (
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
)
SENTENCE_END = (".", "!", "?", "\u2026", "\u06d4", "\u061f", "\u0964")  # . ! ? ... ۔ ؟ ।
CLAUSE_END = (",", ";", ":", "\u060c", "\u061b")                         # , ; : ، ؛
CLOSERS = " \"'\u00bb\u201d)]}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def fmt_clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def progress(label, done, total, t0):
    frac = min(1.0, done / total) if total else 0.0
    elapsed = time.time() - t0
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0
    filled = int(round(30 * frac))
    bar = "#" * filled + "-" * (30 - filled)
    sys.stdout.write(
        f"\r{label}: [{bar}] {frac * 100:5.1f}% | elapsed {fmt_clock(elapsed)} | ETA {fmt_clock(eta)}   "
    )
    sys.stdout.flush()


def srt_time(seconds):
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def txt_time(seconds):
    return srt_time(seconds).replace(",", ".")[:-1]  # hh:mm:ss.xx


def wrap_lines(text, width=42):
    """Break one long line into two balanced lines (broadcast style)."""
    text = text.strip()
    if len(text) <= width:
        return text
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid + 1)
    right = text.find(" ", mid)
    cands = [i for i in (left, right) if i > 0]
    if not cands:
        return text
    cut = min(cands, key=lambda i: abs(i - mid))
    return text[:cut].strip() + "\n" + text[cut + 1:].strip()


def looks_hallucinated(text):
    """Catch looping output such as the same phrase repeated many times."""
    words = text.lower().split()
    return len(words) >= 8 and len(set(words)) / len(words) < 0.3


# ---------------------------------------------------------------------------
# Engine: faster-whisper (CTranslate2) with automatic CUDA -> CPU fallback
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, model_name, device="auto", compute_type=None):
        self.model_name = model_name
        self.compute_type_override = compute_type
        self.model = None
        self.device = None
        self._load("cpu" if device == "cpu" else "cuda")

    def _load(self, device):
        compute = self.compute_type_override or ("float16" if device == "cuda" else "int8")
        print(f"Loading model '{self.model_name}' on {device} ({compute})...")
        try:
            self.model = WhisperModel(self.model_name, device=device, compute_type=compute)
            self.device = device
        except Exception as e:
            if device == "cuda":
                print(f"  GPU not usable ({type(e).__name__}: {e}).\n  Falling back to CPU (int8). This will be slower.")
                self._load("cpu")
            else:
                raise

    def transcribe(self, audio, on_segment=None, **kwargs):
        """Run a transcription and fully consume the generator.

        CUDA problems often only show up at the first real inference call,
        so the fallback to CPU is handled here as well as at load time.
        """
        for attempt in (0, 1):
            try:
                seg_iter, info = self.model.transcribe(audio, **kwargs)
                segments = []
                for seg in seg_iter:
                    segments.append(seg)
                    if on_segment:
                        on_segment(seg)
                return segments, info
            except Exception as e:
                if self.device == "cuda" and attempt == 0:
                    print(f"\n  GPU inference failed ({type(e).__name__}: {e}).\n  Retrying on CPU (int8)...")
                    self._load("cpu")
                    continue
                raise


# ---------------------------------------------------------------------------
# Step 1: transcribe in the source language, with word timestamps
# ---------------------------------------------------------------------------

def pseudo_words(seg):
    """Fallback when a segment has no word timings: spread time by characters."""
    words = seg.text.split()
    total = sum(len(w) for w in words)
    if not total:
        return []
    out, t = [], seg.start
    span = seg.end - seg.start
    for w in words:
        d = span * len(w) / total
        out.append({"start": t, "end": t + d, "text": w})
        t += d
    return out


def transcribe_source(engine, audio, language, beam_size):
    duration = len(audio) / SAMPLE_RATE
    t0 = time.time()
    segments, info = engine.transcribe(
        audio,
        on_segment=lambda s: progress("Transcribing", s.end, duration, t0),
        language=None if language == "auto" else language,
        task="transcribe",
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        condition_on_previous_text=False,
    )
    progress("Transcribing", duration, duration, t0)
    print()

    tokens = []
    for seg in segments:
        if seg.no_speech_prob > 0.9 and seg.avg_logprob < -1.0:
            continue  # almost certainly not speech
        if seg.words:
            for w in seg.words:
                txt = w.word.strip()
                if txt:
                    tokens.append({"start": w.start, "end": w.end, "text": txt})
        else:
            tokens.extend(pseudo_words(seg))
    return tokens, info.language


# ---------------------------------------------------------------------------
# Step 2: build subtitle-sized chunks from REAL word timings
# ---------------------------------------------------------------------------

def _make_chunk(toks):
    return {
        "start": toks[0]["start"],
        "end": toks[-1]["end"],
        "text": " ".join(t["text"] for t in toks),
        "en": "",
    }


def build_chunks(tokens, max_chars, max_duration, pause, min_chunk):
    chunks, cur = [], []

    def flush():
        if cur:
            chunks.append(_make_chunk(cur))
            cur.clear()

    for tok in tokens:
        if cur:
            gap = tok["start"] - cur[-1]["end"]
            new_len = len(" ".join(t["text"] for t in cur)) + 1 + len(tok["text"])
            too_long_time = tok["end"] - cur[0]["start"] > max_duration
            if gap > pause or new_len > max_chars or too_long_time:
                flush()
        cur.append(tok)

        word = tok["text"].rstrip(CLOSERS)
        text_len = len(" ".join(t["text"] for t in cur))
        if word.endswith(SENTENCE_END):
            flush()
        elif word.endswith(CLAUSE_END) and text_len >= max_chars * 0.5:
            flush()
    flush()

    # Merge fragments that are too short to read.
    def short(c):
        return (c["end"] - c["start"]) < min_chunk

    merged = []
    for c in chunks:
        if merged and (short(c) or short(merged[-1])):
            p = merged[-1]
            combined_len = len(p["text"]) + 1 + len(c["text"])
            combined_dur = c["end"] - p["start"]
            if (c["start"] - p["end"] <= pause
                    and combined_len <= max_chars * 1.2
                    and combined_dur <= max_duration * 1.2):
                p["end"] = c["end"]
                p["text"] = p["text"] + " " + c["text"]
                continue
        merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Step 3: translate each chunk from its own audio slice (guaranteed 1:1 pairing)
# ---------------------------------------------------------------------------

def translate_chunks(engine, audio, chunks, language, beam_size, pad):
    total = len(chunks)
    t0 = time.time()
    for i, c in enumerate(chunks, start=1):
        s = max(0, int((c["start"] - pad) * SAMPLE_RATE))
        e = min(len(audio), int((c["end"] + pad) * SAMPLE_RATE))
        clip = audio[s:e]
        if len(clip) >= SAMPLE_RATE * 0.3:
            segs, _ = engine.transcribe(
                clip,
                language=language,
                task="translate",
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(sg.text.strip() for sg in segs).strip()
            c["en"] = "" if looks_hallucinated(text) else text
        progress("Translating ", i, total, t0)
    print()


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_srt(path, cues):
    """cues: list of (start, end, text). Numbering is always sequential."""
    with open(path, "w", encoding="utf-8") as f:
        for n, (start, end, text) in enumerate(cues, start=1):
            f.write(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")


def write_outputs(chunks, base, lang, rtl_marks=False):
    # Stop cues from overlapping the next one.
    for i, c in enumerate(chunks[:-1]):
        nxt = chunks[i + 1]["start"]
        if c["end"] > nxt and nxt > c["start"]:
            c["end"] = nxt

    def orig(c):
        lines = wrap_lines(c["text"])
        if rtl_marks:
            lines = "\n".join("\u202b" + ln + "\u202c" for ln in lines.split("\n"))
        return lines

    bilingual = []
    for c in chunks:
        text = orig(c)
        if c["en"]:
            text += "\n" + wrap_lines(c["en"])
        bilingual.append((c["start"], c["end"], text))
    original_only = [(c["start"], c["end"], orig(c)) for c in chunks]
    english_only = [(c["start"], c["end"], wrap_lines(c["en"])) for c in chunks if c["en"]]

    paths = {
        "bilingual": f"{base}.bilingual.srt",
        "original": f"{base}.{lang}.srt",
        "english": f"{base}.en.srt",
        "txt": f"{base}.txt",
    }
    _write_srt(paths["bilingual"], bilingual)
    _write_srt(paths["original"], original_only)
    _write_srt(paths["english"], english_only)
    with open(paths["txt"], "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"[{txt_time(c['start'])} --> {txt_time(c['end'])}]  {c['text']}\n")
            if c["en"]:
                f.write(f"    {c['en']}\n")
    return paths


# ---------------------------------------------------------------------------
# Per-file pipeline (a GUI or queue can call this directly)
# ---------------------------------------------------------------------------

def process_file(engine, path, out_dir, args):
    name = os.path.splitext(os.path.basename(path))[0]
    base = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {os.path.basename(path)} ===")
    print("Decoding audio...")
    audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE * 0.5:
        raise RuntimeError("audio is empty or too short")

    tokens, lang = transcribe_source(engine, audio, args.language, args.beam_size)
    if not tokens:
        raise RuntimeError("no speech detected")
    print(f"Language: {lang}")

    chunks = build_chunks(tokens, args.max_chars, args.max_duration, args.pause, args.min_chunk)
    print(f"{len(tokens)} words -> {len(chunks)} subtitle chunks")

    if lang == "en":
        for c in chunks:
            c["en"] = ""  # already English; skip the translate pass
    else:
        translate_chunks(engine, audio, chunks, lang, args.beam_size, args.pad)

    paths = write_outputs(chunks, base, lang, args.rtl_marks)
    print("Saved:")
    for p in paths.values():
        print(f"  {p}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_files(args):
    files = list(args.files)
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: folder '{args.folder}' not found.")
            sys.exit(1)
        for f in sorted(os.listdir(args.folder)):
            if f.lower().endswith(MEDIA_EXTS):
                files.append(os.path.join(args.folder, f))
    return files


def pick_local_files():
    found = sorted(f for f in os.listdir(".") if f.lower().endswith(MEDIA_EXTS))
    if not found:
        raw = input("No media files in this folder. Enter a file path: ").strip().strip('"')
        return [raw] if raw else []
    print("Media files in this folder:")
    for i, name in enumerate(found, 1):
        print(f"  [{i}] {name} ({os.path.getsize(name) / (1024 * 1024):.1f} MB)")
    raw = input("Select numbers (e.g. 1,3), or type a path: ").strip().strip('"')
    picked = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(found):
            picked.append(found[int(part) - 1])
        elif part:
            picked.append(part)
    return picked


def parse_args():
    p = argparse.ArgumentParser(description="Urdu/Hindi/Arabic media -> bilingual SRT (original + English).")
    p.add_argument("files", nargs="*", help="audio/video file(s)")
    p.add_argument("--folder", help="process every media file in this folder")
    p.add_argument("--output-dir", help="where to write outputs (default: next to each input file)")
    p.add_argument("--language", default="ur", help="source language code: ur, hi, ar, ... or 'auto' (default: ur)")
    p.add_argument("--model", default="large-v3", help="tiny, base, small, medium, large-v3, or a model path (default: large-v3)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="auto tries the GPU first, then CPU")
    p.add_argument("--compute-type", help="override, e.g. float16, int8_float16, int8")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=84, help="max characters per subtitle chunk (default 84 = two 42-char lines)")
    p.add_argument("--max-duration", type=float, default=7.0, help="max seconds per chunk")
    p.add_argument("--pause", type=float, default=0.6, help="a silence longer than this (s) starts a new chunk")
    p.add_argument("--min-chunk", type=float, default=1.0, help="chunks shorter than this (s) get merged with a neighbour")
    p.add_argument("--pad", type=float, default=0.15, help="seconds of audio context added around each chunk when translating")
    p.add_argument("--rtl-marks", action="store_true", help="wrap original lines in Unicode RTL marks (try if Urdu/Arabic shows reversed)")
    args = p.parse_args()
    if args.max_chars < 10:
        p.error("--max-chars must be at least 10")
    return args


def main():
    args = parse_args()
    files = collect_files(args)
    jobs = [("file", f) for f in files]

    if not jobs:
        jobs = [("file", f) for f in pick_local_files()]
    if not jobs:
        print("Nothing to do.")
        sys.exit(1)

    engine = Engine(args.model, args.device, args.compute_type)

    done, failed = [], []
    for kind, value in jobs:
        try:
            if kind == "file":
                if not os.path.isfile(value):
                    raise FileNotFoundError(f"'{value}' not found")
                path = value
                out_dir = args.output_dir or os.path.dirname(os.path.abspath(value))
            process_file(engine, path, out_dir, args)
            done.append(value)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nFAILED: {value}\n  {type(e).__name__}: {e}")
            failed.append(value)

    print(f"\nFinished: {len(done)} succeeded, {len(failed)} failed.")
    for v in failed:
        print(f"  failed: {v}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
