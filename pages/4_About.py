"""4_About.py
Streamlit page that provides a comprehensive project overview, methodology, and disclaimer.
"""
import streamlit as st

def run():
    st.title("🩺 About PneumoScan")
    st.markdown("""
    **Project Name:** PneumoScan
    
    **Organization:** Apex Medical Imaging Solutions
    
    **Objective:**
    - Build a CNN‑based prototype that classifies chest X‑ray images as **NORMAL** or **PNEUMONIA**.
    - Provide an interactive UI for researchers and students to explore model predictions and performance.
    
    **Problem Statement**
    Pneumonia is a leading cause of mortality worldwide, especially among children and the elderly. Early detection via chest radiographs can accelerate treatment, but manual interpretation is time‑consuming and requires expertise.
    
    **Proposed Solution**
    A deep‑learning model trained on the publicly available *Chest X‑Ray Images (Pneumonia)* dataset. The application demonstrates:
    1. End‑to‑end data loading & preprocessing.
    2. Custom CNN architecture (and optional MobileNetV2 transfer learning).
    3. Training with early stopping, checkpointing, and LR scheduling.
    4. Comprehensive evaluation metrics (accuracy, precision, recall, specificity, F1‑score, ROC‑AUC).
    5. A Streamlit dashboard with image upload, Grad‑CAM visualisation, and performance plots.
    
    **CNN Architecture**
    - 3 convolutional blocks (32, 64, 128 filters) with BatchNorm & ReLU.
    - Max‑pooling after each block.
    - Global average pooling → Dense(128) → Dropout(0.5) → Sigmoid output.
    
    **Dataset Details**
    - Source: [Kaggle – Chest X‑Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - License: CC‑0 (public domain).
    - Structure: `train/`, `val/`, `test/` each containing `NORMAL/` and `PNEUMONIA/` folders.
    - No data leakage: test set is untouched until final evaluation.
    
    **Technologies Used**
    - Python 3.10
    - TensorFlow / Keras
    - OpenCV, Pillow, NumPy, Pandas, Matplotlib, Seaborn, Scikit‑learn
    - Streamlit for the UI
    
    **Limitations**
    - The model is trained on a limited dataset collected from a single institution.
    - No clinical validation; performance metrics are only indicative.
    - Grad‑CAM visualisation is approximate and not a substitute for radiologist interpretation.
    
    **Future Work**
    - Expand training data with multi‑center datasets.
    - Perform external validation and calibration of probabilities.
    - Incorporate explainability methods beyond Grad‑CAM (e.g., SHAP, LIME).
    - Deploy as a lightweight web service with secure authentication.
    
    **Medical Disclaimer**
    > **This application is an educational Deep Learning prototype and is not a medical diagnostic system.**
    > Its predictions may be incorrect and must not replace professional radiological interpretation or medical advice. Never use this result to make treatment decisions or delay medical care.
    """)
    st.subheader("Technology Stack")
    st.markdown("""
    - **Python** 3.10
    - **TensorFlow / Keras** 2.13
    - **OpenCV**, **Pillow** for image handling
    - **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, **Scikit‑learn** for data processing & evaluation
    - **Streamlit** 1.28 for the interactive dashboard
    """)
    st.image("assets/logo.png", width=200)

