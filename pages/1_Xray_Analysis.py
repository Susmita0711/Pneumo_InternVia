"""1_Xray_Analysis.py
Streamlit page for uploading a chest X‑ray, running inference, and (optionally) displaying a Grad‑CAM heatmap.
"""
import streamlit as st
import os
import datetime
from src.predict import predict_image
from src.gradcam import generate_gradcam
from src.preprocessing import preprocess_path

# Constants
MODEL_CHOICES = {"Custom CNN": "cnn", "MobileNetV2 (Transfer)": "mobilenet"}
LAST_CONV_LAYER = "conv2d_2"  # name from custom CNN; adjust if using MobileNetV2

def _init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

def run():
    st.title("🩺 X‑ray Analysis")
    st.subheader("Upload a chest X‑ray and get a prediction")

    _init_session_state()

    model_option = st.selectbox("Select model", list(MODEL_CHOICES.keys()))
    model_name = MODEL_CHOICES[model_option]

    uploaded = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        # Save uploaded file to a temporary location
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "uploads"))
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Show preview
        st.image(temp_path, caption="Uploaded X‑ray", use_column_width=True)

        # Run prediction
        try:
            label, prob = predict_image(temp_path, model_name=model_name)
            confidence = prob if label == "PNEUMONIA" else 1 - prob
            st.success(f"**Prediction:** {label}")
            st.info(f"**Model confidence** (un‑calibrated): {confidence:.2%}")

            # Record in history
            st.session_state.history.append(
                {
                    "image": temp_path,
                    "label": label,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        # Optional Grad‑CAM
        if st.checkbox("Show Grad‑CAM heatmap (experimental)"):
            try:
                # Load the model for Grad‑CAM (the same checkpoint used by predict_image)
                from src.predict import load_model
                model = load_model(model_name)
                heatmap_path = generate_gradcam(
                    image_path=temp_path,
                    model=model,
                    last_conv_layer_name=LAST_CONV_LAYER,
                    output_dir=os.path.join(os.path.dirname(__file__), "..", "assets", "gradcam"),
                )
                st.image(heatmap_path, caption="Grad‑CAM overlay", use_column_width=True)
            except Exception as exc:
                st.warning(f"Grad‑CAM could not be generated: {exc}")
    else:
        st.info("Upload an image to run analysis.")

