
import streamlit as st
import numpy as np
import cv2
import pydicom
import gdown
import os
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = "pulmoscan_model.keras"
GDRIVE_FILE_ID = "1ZrQm_T7_lbax1qnh_d-_By7BFy6OoaQ6"
#https://drive.google.com/file/d/1ZrQm_T7_lbax1qnh_d-_By7BFy6OoaQ6/view?usp=sharing

@st.cache_resource
def load_pulmoscan_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_pulmoscan_model()

def preprocess_image(uploaded_file):
    if uploaded_file.name.lower().endswith(".dcm"):
        dcm = pydicom.dcmread(uploaded_file)
        img = dcm.pixel_array.astype("float32")
    else:
        img = np.array(Image.open(uploaded_file).convert("L")).astype("float32")

    # Properly rescale actual pixel range to 0-255 for correct display,
    # regardless of the original bit depth (8-bit, 12-bit, or 16-bit DICOM)
    img_display = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype("uint8")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img_3ch = np.stack([img, img, img], axis=-1)
    return np.expand_dims(img_3ch, axis=0), img_display

st.title("PulmoScan — Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image (DICOM, JPG, or PNG) to receive an automated pneumonia screening prediction. This tool is intended as a decision-support aid and does not replace professional medical diagnosis.")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["dcm", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_batch, img_display = preprocess_image(uploaded_file)
    st.image(img_display, caption="Uploaded X-Ray", clamp=True, use_column_width=True)

    prob = float(model.predict(img_batch, verbose=0)[0][0])
    label = "Pneumonia Detected" if prob > 0.5 else "No Pneumonia Detected"
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
