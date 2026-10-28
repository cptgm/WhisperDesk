"""WhisperDesk - one page, two modes. Run:  streamlit run app.py"""
import importlib.util

import streamlit as st

st.set_page_config(page_title="WhisperDesk | AI Subtitles", layout="wide")

from whisperdesk_common import mode_selector, render_hero, run_app

render_hero()

mode = mode_selector()

if mode == "online":
    missing = [m for m in ("deep_translator", "requests") if importlib.util.find_spec(m) is None]
    if missing:
        st.error("Online mode needs extra packages that are not installed. "
                 "Run: pip install -r requirements_all.txt")
        st.stop()

run_app(mode, configure=False, show_title=False)
