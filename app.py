"""app.py
Main Streamlit entry point for PneumoScan.
It loads the selected page based on the URL query parameter ``page``.
If no page is specified, the Home page is shown.
"""
import streamlit as st
import os

# Set page config (branding)
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("PneumoScan")
pages = {
    "Home": "pages/1_Xray_Analysis.py",
    "X‑ray Analysis": "pages/1_Xray_Analysis.py",
    "Model Performance": "pages/2_Model_Performance.py",
    "Prediction History": "pages/3_Prediction_History.py",
    "About": "pages/4_About.py",
}
selected = st.sidebar.radio("Navigate", list(pages.keys()))

# Dynamically import and run the selected page
page_path = pages[selected]
# Ensure the path is relative to this file
module_name = os.path.splitext(os.path.basename(page_path))[0]
import importlib.util
spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), page_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Each page module must expose a ``run()`` function
if hasattr(module, "run"):
    module.run()
else:
    st.error(f"Page {selected} does not implement a run() function.")

