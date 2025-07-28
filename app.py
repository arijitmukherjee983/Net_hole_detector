import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

from net_hole_detector.model import predict
from net_hole_detector.utils import load_model

# Load model once using Streamlit cache
model = load_model()

st.title("🎣 Fishing Net Hole Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_pil.save(tmp_file.name)
        temp_path = tmp_file.name

    detections, annotated_image = predict(model, image_pil, temp_path)

    st.image(annotated_image, caption="Detected Net Holes", use_container_width=True)
    os.remove(temp_path)
