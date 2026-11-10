"""3_Prediction_History.py
Streamlit page showing the session‑based prediction history.
"""
import streamlit as st
import os
from datetime import datetime

def _display_history():
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions made in this session.")
        return
    # Show as a table with thumbnails
    for entry in reversed(history):
        cols = st.columns([1, 2, 2, 2])
        with cols[0]:
            st.image(entry["image"], width=80)
        with cols[1]:
            st.markdown(f"**Label:** {entry['label']}")
        with cols[2]:
            st.markdown(f"**Confidence:** {entry['confidence']:.2%}")
        with cols[3]:
            st.markdown(f"**Time:** {entry['timestamp']}")

def run():
    st.title("📜 Prediction History")
    st.subheader("Session‑based records of analysed X‑rays")
    _display_history()
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared.")

