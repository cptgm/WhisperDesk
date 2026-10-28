"""WhisperDesk shared app logic.

Do not run this file directly. Run one of:
    app_local.py   -> fully offline (Whisper translates to English on your PC)
    app_online.py  -> speech recognition is local, text translation uses an online service (internet)

Speech recognition always runs locally through faster-whisper (CTranslate2) using
the helpers in translate_local.py.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av

# PyAV compatibility: newer PyAV removed the metadata_errors argument faster-whisper passes.
if not getattr(av.open, "_wd_patched", False):
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    _safe_av_open._wd_patched = True
    av.open = _safe_av_open

import pandas as pd
import streamlit as st
from faster_whisper import decode_audio

import translate_local as core

# ---------------------------------------------------------------------------
# Languages. Whisper codes and Google codes are kept separate on purpose
# (e.g. Chinese is "zh" for Whisper but "zh-CN" for Google).
# ---------------------------------------------------------------------------

SOURCE_LANGUAGES = {
    "Auto-detect spoken language": "auto",
    "Urdu": "ur", "Hindi": "hi", "English": "en", "Arabic": "ar",
    "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Russian": "ru", "Chinese": "zh",
    "Japanese": "ja", "Korean": "ko", "Turkish": "tr", "Persian": "fa",
    "Bengali": "bn", "Punjabi": "pa", "Indonesian": "id", "Dutch": "nl",
    "Polish": "pl", "Thai": "th", "Vietnamese": "vi", "Swahili": "sw",
}

ONLINE_TARGETS = {
    "English": "en", "Urdu": "ur", "Hindi": "hi", "Arabic": "ar",
    "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Russian": "ru", "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja", "Korean": "ko", "Turkish": "tr", "Persian": "fa",
    "Bengali": "bn", "Punjabi": "pa", "Indonesian": "id", "Dutch": "nl",
    "Polish": "pl", "Thai": "th", "Vietnamese": "vi", "Swahili": "sw",
}

LOCAL_TARGETS = {"English": "en"}  # Whisper's built-in translate task only outputs English

LANGUAGE_NAMES = {
    "en": "English", "ur": "Urdu", "hi": "Hindi", "ar": "Arabic",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "fa": "Persian",
    "bn": "Bengali", "pa": "Punjabi", "id": "Indonesian", "nl": "Dutch",
    "pl": "Polish", "th": "Thai", "vi": "Vietnamese", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "sd": "Sindhi", "ps": "Pashto",
    "ne": "Nepali", "si": "Sinhala", "ms": "Malay", "tl": "Tagalog",
    "el": "Greek", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "sv": "Swedish",
    "uk": "Ukrainian", "he": "Hebrew",
}

RTL_LANGS = {"ur", "ar", "fa", "he", "ps", "sd"}

MEDIA_TYPES = ["mp3", "wav", "m4a", "flac", "aac", "ogg", "opus", "wma",
               "mp4", "mkv", "mov", "avi", "webm", "flv", "wmv", "m4v"]
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}

QUALITY_MAP = {"Base": "base", "Small": "small", "Medium": "medium", "Large-V3": "large-v3"}
# Turbo is transcription-only, so it is offered in online mode only (offline mode needs Whisper's translate task).
QUALITY_MAP_ONLINE = {**QUALITY_MAP, "Turbo (fast)": "large-v3-turbo"}

THEMES = {
    "local": dict(
        accent="#10b981", dark="#059669", icon="🔒", title="Fully offline",
        tagline="Nothing leaves this PC once the model is downloaded. Subtitles are translated to English only.",
    ),
    "online": dict(
        accent="#6366f1", dark="#4f46e5", icon="🌐", title="Online translation",
        tagline="Your audio stays on this PC, but the subtitle <b>text</b> is sent to an online translation "
                "service. Do not use this mode for confidential material.",
    ),
}
FILE_COLORS = {"original": "#f59e0b", "translated": "#0ea5e9", "bilingual": "#8b5cf6", "transcript": "#f43f5e"}
MODE_LABELS = {"local": "🔒 Offline · private, English only", "online": "🌐 Online · 23 languages"}


BASE_CSS = """
<style>
.stApp, .stApp p, .stApp label, .stApp li { font-family: "Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif; }
.block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 4rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; border-color: rgba(255,255,255,.09);
  background: rgba(255,255,255,.025); }
button[data-testid^="stBaseButton"] { border-radius: 10px; font-weight: 600; letter-spacing: .01em; }
[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { border-radius: 10px; }
h4 { letter-spacing: -.01em; margin-top: 1.6rem; }
.wd-hero { margin: 0 0 1.2rem; text-align: center; }
.wd-hero h1 { font-family: "Segoe UI Variable Display","Segoe UI",system-ui,sans-serif; font-size: 3.4rem;
  font-weight: 700; letter-spacing: -.035em; line-height: 1.05; margin: 0 0 .6rem; padding: 0; }
.wd-hero p.lead { font-size: 1.15rem; opacity: .75; margin: 0 auto; max-width: 34rem; }
.st-key-wd_mode_v2 {
  display: flex !important; flex-direction: column !important; align-items: center !important;
  align-self: center !important; margin-left: auto !important; margin-right: auto !important;
  text-align: center !important;
}
.st-key-wd_mode_v2 [data-testid="stButtonGroup"] {
  display: flex !important; flex-direction: column !important; align-items: center !important;
}
.st-key-wd_mode_v2 [data-testid="stWidgetLabel"] {
  justify-content: center !important; text-align: center !important; width: 100% !important;
}
.wd-mode { border: 1px solid var(--wd-accent, #6366f1); border-color: color-mix(in srgb, var(--wd-accent, #6366f1) 45%, transparent);
  background: linear-gradient(180deg, var(--wd-tint, #6366f122), transparent); border-radius: 14px;
  padding: .8rem 1.2rem; margin: .8rem auto 1.4rem; max-width: 640px; text-align: center; }
.wd-mode b { font-size: 1.05rem; }
.wd-mode span.beta { font-size: .75rem; opacity: .6; margin-left: .5rem; }
.wd-mode div { opacity: .85; font-size: .93rem; margin-top: .25rem; }
@media (max-width: 640px) { .wd-hero h1 { font-size: 2.4rem; } }
</style>
"""


def render_hero():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='wd-hero'><h1>WhisperDesk</h1>"
        "<p class='lead'>Turn speech into timed, bilingual subtitles. Your audio never leaves this PC.</p></div>",
        unsafe_allow_html=True,
    )


def theme_css(accent: str, dark: str) -> str:
    files = "".join(
        f".st-key-dl_{k} button{{border-left:5px solid {c} !important;}}" for k, c in FILE_COLORS.items()
    )
    return (
        "<style>"
        f":root{{--wd-accent:{accent};--wd-tint:{accent}26;}}"
        f"button[data-testid='stBaseButton-primary']{{background-color:{accent};border-color:{accent};}}"
        f"button[data-testid='stBaseButton-primary']:hover{{background-color:{dark};border-color:{dark};}}"
        f"button[data-testid='stBaseButton-segmented_controlActive'],button[kind='segmented_controlActive']{{background-color:{accent}26;"
        f"border-color:{accent};color:{accent};font-weight:600;}}"
        f"{files}</style>"
    )


def mode_selector() -> str:
    """Segmented mode switch for the combined app. Returns 'local' or 'online'."""
    busy = st.session_state.get("is_generating", False)
    pick = st.segmented_control(
        "Translation mode", options=list(MODE_LABELS), format_func=lambda k: MODE_LABELS[k],
        default="local", selection_mode="single", key="wd_mode_v2", disabled=busy,
        help="Speech recognition always runs on this PC. Online mode sends the subtitle text to a translation service.",
    )
    return pick or "local"


def base_code(code: str) -> str:
    return (code or "").lower().split("-")[0]


def get_full_language_name(code: str) -> str:
    if not code:
        return "Unknown"
    return LANGUAGE_NAMES.get(base_code(code), code.capitalize())


def has_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@st.cache_resource(show_spinner="Loading speech-recognition model…")
def load_engine(model_name: str, use_gpu: bool):
    # Engine falls back from CUDA to CPU by itself if the GPU turns out to be unusable.
    return core.Engine(model_name, device="auto" if use_gpu else "cpu")


# ---------------------------------------------------------------------------
# Output builders (pure functions)
# ---------------------------------------------------------------------------

def fix_overlaps(chunks):
    """Return copies of chunks where no cue runs into the next one."""
    out = [dict(c) for c in chunks]
    for i in range(len(out) - 1):
        nxt = out[i + 1]["start"]
        if out[i]["end"] > nxt > out[i]["start"]:
            out[i]["end"] = nxt
    return out


def _fmt(text: str, rtl: bool) -> str:
    text = core.wrap_lines((text or "").strip())
    if rtl:
        text = "\n".join("\u202b" + line + "\u202c" for line in text.split("\n"))
    return text


def build_srt(chunks, mode: str, orig_rtl: bool = False, trans_rtl: bool = False) -> bytes:
    """mode: 'original' | 'translated' | 'bilingual'. Cue numbers are always sequential."""
    cues = []
    for c in chunks:
        orig = (c.get("text") or "").strip()
        trans = (c.get("translated") or "").strip()
        if mode == "original":
            text = _fmt(orig, orig_rtl) if orig else ""
        elif mode == "translated":
            text = _fmt(trans, trans_rtl) if trans else ""
        else:
            parts = []
            if orig:
                parts.append(_fmt(orig, orig_rtl))
            if trans and trans != orig:
                parts.append(_fmt(trans, trans_rtl))
            text = "\n".join(parts)
        if text:
            cues.append((c["start"], c["end"], text))
    blocks = [
        f"{n}\n{core.srt_time(s)} --> {core.srt_time(e)}\n{t}"
        for n, (s, e, t) in enumerate(cues, start=1)
    ]
    return ("\n\n".join(blocks) + "\n").encode("utf-8-sig")


def build_transcript(chunks) -> bytes:
    lines = [
        f"[{core.txt_time(c['start'])} --> {core.txt_time(c['end'])}]  {c['text'].strip()}"
        for c in chunks if (c.get("text") or "").strip()
    ]
    return ("\n".join(lines) + "\n").encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Editable table helpers (pure functions)
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    return core.srt_time(seconds).replace(",", ".")  # HH:MM:SS.mmm


def parse_time(value) -> float:
    """Accepts 'HH:MM:SS.mmm', 'HH:MM:SS,mmm', 'MM:SS.m' or plain seconds."""
    s = str(value if value is not None else "").strip().replace(",", ".")
    if not s:
        raise ValueError("empty time")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError("too many ':'")
    secs = 0.0
    for part in parts:
        secs = secs * 60 + float(part)
    if not math.isfinite(secs) or secs < 0:
        raise ValueError("invalid time")
    return secs


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def chunks_to_df(chunks, same_lang: bool) -> "pd.DataFrame":
    rows = []
    for c in chunks:
        row = {"Start": fmt_time(c["start"]), "End": fmt_time(c["end"]), "Original": c["text"]}
        if not same_lang:
            row["Translation"] = c.get("translated", "")
        rows.append(row)
    cols = ["Start", "End", "Original"] + ([] if same_lang else ["Translation"])
    return pd.DataFrame(rows, columns=cols)


def df_to_chunks(df, same_lang: bool):
    """Turn the edited table back into chunks. Returns (chunks, list_of_row_errors)."""
    chunks, errors = [], []
    for i, row in enumerate(df.to_dict("records"), start=1):
        orig = _cell(row.get("Original"))
        trans = orig if same_lang else _cell(row.get("Translation"))
        if not orig and not trans:
            continue  # blank row
        try:
            start, end = parse_time(row.get("Start")), parse_time(row.get("End"))
        except ValueError:
            errors.append(f"Row {i} (start '{_cell(row.get('Start'))}'): time not understood")
            continue
        if end <= start:
            errors.append(f"Row {i} (start '{_cell(row.get('Start'))}'): End must be after Start")
            continue
        chunks.append({"start": start, "end": end, "text": orig, "translated": trans})
    chunks.sort(key=lambda c: c["start"])
    return fix_overlaps(chunks), errors


# ---------------------------------------------------------------------------
# Translation back-ends
# ---------------------------------------------------------------------------

def translate_offline(engine, audio, chunks, lang, beam_size, on_progress, pad=0.15):
    """Whisper translate task on each chunk's own audio slice (English only, fully local).

    Returns (chunks_with_translated, failed_count).
    """
    total, failed, out = len(chunks), 0, []
    for i, c in enumerate(chunks, start=1):
        s = max(0, int((c["start"] - pad) * core.SAMPLE_RATE))
        e = min(len(audio), int((c["end"] + pad) * core.SAMPLE_RATE))
        clip = audio[s:e]
        text = ""
        if len(clip) >= core.SAMPLE_RATE * 0.3:
            segs, _ = engine.transcribe(
                clip, language=lang, task="translate", beam_size=beam_size,
                vad_filter=False, condition_on_previous_text=False, without_timestamps=True,
            )
            text = " ".join(sg.text.strip() for sg in segs).strip()
            if core.looks_hallucinated(text):
                text = ""
        item = dict(c)
        item["translated"] = text
        if not text:
            failed += 1
        out.append(item)
        on_progress(i, total)
    return out, failed


def google_translate(text: str, target_code: str) -> str:
    """Returns the translation, or "" if every attempt failed (never returns the input)."""
    text = (text or "").strip()
    if not text:
        return ""
    for _attempt in range(2):
        try:
            from deep_translator import GoogleTranslator
            out = GoogleTranslator(source="auto", target=target_code).translate(text)
            if out:
                return out
        except Exception:
            pass
        try:
            import requests
            resp = requests.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": "auto", "tl": target_code, "dt": "t", "q": text},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                out = "".join(seg[0] for seg in data[0] if seg and seg[0])
                if out:
                    return out
        except Exception:
            pass
        time.sleep(0.5)
    return ""


def translate_online(chunks, target_code, on_progress):
    """Parallel Google translation (4 workers). Returns (chunks_with_translated, failed_count)."""
    total, failed, out = len(chunks), 0, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(google_translate, c["text"], target_code) for c in chunks]
        for i, (c, fut) in enumerate(zip(chunks, futures), start=1):
            text = fut.result()
            item = dict(c)
            item["translated"] = text
            if not text:
                failed += 1
            out.append(item)
            on_progress(i, total)
    return out, failed


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def run_app(mode: str, configure: bool = True, show_title: bool = True):
    assert mode in ("local", "online")
    is_online = mode == "online"
    targets = ONLINE_TARGETS if is_online else LOCAL_TARGETS

    if configure:
        st.set_page_config(page_title="WhisperDesk | AI Subtitles", layout="wide")

    for key, default in (("is_generating", False), ("subtitles_data", None), ("last_error", None), ("editor_version", 0)):
        if key not in st.session_state:
            st.session_state[key] = default

    def on_generate_click():
        st.session_state.is_generating = True
        st.session_state.last_error = None
        st.session_state.subtitles_data = None
        st.session_state.editor_version += 1

    def undo_edits():
        st.session_state.editor_version += 1

    def reset_generation():
        st.session_state.subtitles_data = None
        st.session_state.last_error = None
        st.session_state.is_generating = False

    is_locked = st.session_state.is_generating
    gpu = has_cuda()

    theme = THEMES[mode]
    accent = theme["accent"]
    st.markdown(theme_css(accent, theme["dark"]), unsafe_allow_html=True)
    if show_title:
        render_hero()
    st.markdown(
        f"<div class='wd-mode'><b>{theme['icon']} {theme['title']}</b><span class='beta'>Beta</span>"
        f"<div>{theme['tagline']}</div></div>",
        unsafe_allow_html=True,
    )

    if is_locked:
        st.info("🔒 **Subtitles are currently generating.** Settings are locked to prevent interruptions.")

    with st.container(border=True):
        uploaded = st.file_uploader(
            "1. Upload an audio or video file", type=MEDIA_TYPES, disabled=is_locked,
            help="Supported: MP4, MKV, MOV, MP3, WAV, M4A, WebM, and more.",
        )
        if uploaded is not None:
            ext = Path(uploaded.name).suffix.lower()
            with st.expander(f"▶️ Preview Media: {uploaded.name} ({uploaded.size / (1024 * 1024):.1f} MB)"):
                if ext in VIDEO_EXTS:
                    st.video(uploaded)
                else:
                    st.audio(uploaded)

        col1, col2 = st.columns(2)
        with col1:
            source_label = st.selectbox(
                "2. Spoken language", list(SOURCE_LANGUAGES.keys()), index=0, disabled=is_locked,
                help="For Urdu or mixed Urdu/Hindi speech, select 'Urdu' directly to get Urdu (Nastaliq) text.",
            )
        with col2:
            target_label = st.selectbox(
                "3. Subtitle language", list(targets.keys()), index=0, disabled=is_locked,
                help=None if is_online else "The offline version translates to English only.",
            )

        col3, col4 = st.columns(2)
        with col3:
            qmap = QUALITY_MAP_ONLINE if is_online else QUALITY_MAP
            quality_label = st.selectbox(
                "Speech recognition quality", list(qmap.keys()),
                index=3 if gpu else 1, disabled=is_locked,
                help="Large-V3 is the most accurate (important for Urdu/Hindi) but slow without an NVIDIA GPU."
                     + (" Turbo is much faster, with slightly lower accuracy." if is_online else ""),
            )
            model_name = qmap[quality_label]
        with col4:
            st.write("")
            st.write("")
            rtl_marks = st.checkbox(
                "Add RTL marks (Urdu/Arabic)", value=False, disabled=is_locked,
                help="Turn on only if Urdu/Arabic text appears reversed or scrambled in your video player.",
            )

        st.caption("Hardware: " + ("NVIDIA GPU detected." if gpu else "no NVIDIA GPU detected, using CPU (large models will be slow)."))

        if is_locked:
            st.button("⏳ Generating subtitles…", disabled=True, use_container_width=True)
        else:
            st.button("✨ Generate subtitles", type="primary", use_container_width=True,
                      disabled=(uploaded is None), on_click=on_generate_click)

    st.caption(
        "Speech recognition runs locally via Faster-Whisper; the first run downloads the selected model. "
        + ("Translation requires internet." if is_online else "Translation also runs locally.")
    )

    # ------------------------------------------------------------------ generate
    if is_locked and uploaded is None:
        st.session_state.is_generating = False
        st.rerun()

    if is_locked and uploaded is not None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix.lower()) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name

            status = st.status("Preparing your subtitles…", expanded=True)
            status.write(f"Loading speech-recognition model ({quality_label})…")
            engine = load_engine(model_name, gpu)
            status.write(f"Running on {engine.device.upper()}.")

            status.write("Reading audio and analyzing speech…")
            audio = decode_audio(temp_path, sampling_rate=core.SAMPLE_RATE)
            if len(audio) < core.SAMPLE_RATE * 0.5:
                raise ValueError("This file is empty or too short to process.")

            tokens, detected = core.transcribe_source(engine, audio, SOURCE_LANGUAGES[source_label], beam_size=5)
            if not tokens:
                raise ValueError("No clear speech was detected. Try a clearer or longer recording.")

            chunks = fix_overlaps(core.build_chunks(tokens, max_chars=84, max_duration=7.0, pause=0.6, min_chunk=1.0))
            det_name = get_full_language_name(detected)
            status.write(f"Speech detected: **{det_name}** ({detected}). Created {len(chunks)} subtitle sections.")

            target_code = targets[target_label]
            warnings = []
            if base_code(detected) == base_code(target_code):
                translated_chunks = [dict(c, translated=c["text"]) for c in chunks]
                status.write("Spoken language matches the subtitle language; no translation needed.")
            else:
                status.write(f"Translating subtitles into **{target_label}**…")
                pbar = st.progress(0, text="Translating subtitles…")

                def on_progress(i, total):
                    pbar.progress(i / total, text=f"Translating subtitles… {i}/{total}")

                if is_online:
                    translated_chunks, failed = translate_online(chunks, target_code, on_progress)
                else:
                    translated_chunks, failed = translate_offline(engine, audio, chunks, detected, 5, on_progress)
                pbar.empty()

                total = len(chunks)
                if failed == total:
                    warnings.append(
                        "**Translation failed for every line.** "
                        + ("Check your internet connection or try again later; the translation service may be limiting requests."
                           if is_online else
                           "Try a larger model, or check that the speech is clear.")
                        + " You can type translations into the table below."
                    )
                elif failed:
                    warnings.append(
                        f"**{failed} of {total} lines could not be translated** and are blank in the Translation "
                        "column. Fill them in below, or they are left out of the translated file."
                    )

            same_lang = base_code(detected) == base_code(target_code)
            status.update(label="Subtitles generated!", state="complete", expanded=False)

            st.session_state.subtitles_data = {
                "stem": Path(uploaded.name).stem,
                "detected": detected,
                "det_name": det_name,
                "target_label": target_label,
                "target_code": target_code,
                "count": len(chunks),
                "same_lang": same_lang,
                "table": chunks_to_df(translated_chunks, same_lang),
                "o_rtl": rtl_marks and base_code(detected) in RTL_LANGS,
                "t_rtl": rtl_marks and base_code(target_code) in RTL_LANGS,
                "warnings": warnings,
            }
        except Exception as exc:
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            st.session_state.is_generating = False
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        st.rerun()  # outside try/finally so the error saved above survives the reload

    # ------------------------------------------------------------------ results
    if st.session_state.last_error:
        st.error(f"Could not generate subtitles: {st.session_state.last_error}")
        st.info("If this is a model download or memory error, try 'Base' or 'Small' and check your internet connection. "
                "If Turbo reports an unknown model, update the engine: pip install -U faster-whisper")

    data = st.session_state.subtitles_data
    if data is not None:
        det, det_name, tcode = data["detected"], data["det_name"], data["target_code"].replace("-", "_")
        st.divider()
        st.success(f"🎉 **Subtitles ready!** Detected **{det_name}** speech and created {data['count']} subtitle sections.")
        for w in data["warnings"]:
            st.warning(w)

        if base_code(det) in {"hi", "ur"}:
            st.info(
                f"💡 **Urdu & Hindi note:** spoken Urdu and Hindi sound almost identical, and the model detected "
                f"**{det_name}**. For Urdu (Nastaliq) text, select **Urdu** as the spoken language and run again."
            )

        same_lang = data["same_lang"]
        st.markdown("#### ✏️ Edit subtitles")
        st.caption(
            "Click any cell to fix text or timing. Times use HH:MM:SS.mmm. Add rows at the bottom of the "
            "table, or delete a row by selecting it and pressing Delete. The downloads below always use "
            "your edited version."
        )
        cfg = {
            "Start": st.column_config.TextColumn("Start", width="small", required=True),
            "End": st.column_config.TextColumn("End", width="small", required=True),
            "Original": st.column_config.TextColumn("Text" if same_lang else "Original", width="large"),
        }
        if not same_lang:
            cfg["Translation"] = st.column_config.TextColumn(data["target_label"], width="large")
        edited = st.data_editor(
            data["table"], key=f"editor_{st.session_state.editor_version}", num_rows="dynamic",
            use_container_width=True, hide_index=True, column_config=cfg,
        )
        edited_chunks, row_errors = df_to_chunks(edited, same_lang)
        if row_errors:
            st.warning("These rows are left out of the downloads until fixed:\n\n"
                       + "\n".join(f"- {e}" for e in row_errors[:8])
                       + (f"\n- …and {len(row_errors) - 8} more" if len(row_errors) > 8 else ""))
        st.button("↩️ Undo all edits", on_click=undo_edits)

        o_rtl, t_rtl = data["o_rtl"], data["t_rtl"]
        f = {
            "original": build_srt(edited_chunks, "original", o_rtl, t_rtl),
            "translated": build_srt(edited_chunks, "translated", o_rtl, t_rtl),
            "bilingual": build_srt(edited_chunks, "bilingual", o_rtl, t_rtl),
            "transcript": build_transcript(edited_chunks),
        }
        stem = data["stem"]
        nothing = not edited_chunks
        if nothing:
            st.error("The table has no valid rows, so there is nothing to download.")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(f"⬇️ {det_name} subtitles (.srt)", f["original"], f"{stem}.{det}.srt",
                               "application/x-subrip", key="dl_original", use_container_width=True,
                               disabled=nothing)
            st.download_button(f"⬇️ Bilingual {det_name} + {data['target_label']} (.srt)", f["bilingual"],
                               f"{stem}.bilingual.srt", "application/x-subrip", key="dl_bilingual",
                               type="primary", use_container_width=True, disabled=nothing)
        with c2:
            st.download_button(f"⬇️ {data['target_label']} subtitles (.srt)", f["translated"],
                               f"{stem}.{tcode}.srt", "application/x-subrip", key="dl_translated",
                               use_container_width=True, disabled=nothing)
            st.download_button(f"⬇️ {det_name} transcript (.txt)", f["transcript"], f"{stem}.{det}.txt",
                               "text/plain", key="dl_transcript", use_container_width=True,
                               disabled=nothing)

        st.markdown("#### 👁️ Preview files")
        names = {
            "original": f"{stem}.{det}.srt",
            "translated": f"{stem}.{tcode}.srt",
            "bilingual": f"{stem}.bilingual.srt",
            "transcript": f"{stem}.{det}.txt",
        }
        labels = {
            "original": f"{det_name} SRT",
            "translated": f"{data['target_label']} SRT",
            "bilingual": "Bilingual SRT",
            "transcript": "Transcript",
        }
        pick = st.segmented_control(
            "Preview file", options=list(labels), format_func=lambda k: labels[k],
            default="bilingual", selection_mode="single", key="wd_preview", label_visibility="collapsed",
        ) or "bilingual"
        color = FILE_COLORS[pick]
        st.markdown(
            f"<div style='border-left:4px solid {color};background:{color}1a;padding:6px 12px;"
            f"border-radius:0 8px 8px 0;margin:6px 0;font-weight:600;color:{color};'>{names[pick]}</div>",
            unsafe_allow_html=True,
        )
        shown = f[pick].decode("utf-8-sig")
        st.code(shown[:12000], language="text")
        if len(shown) > 12000:
            st.caption("Preview shows the first 12,000 characters. The download contains everything.")

        st.button("🔄 Process another file / Reset", on_click=reset_generation, use_container_width=True)

    st.divider()
    st.caption("WhisperDesk: speech recognition runs locally on your device. Always review subtitles before publishing.")
