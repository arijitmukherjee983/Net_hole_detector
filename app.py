import streamlit as st
import numpy as np
import cv2
from roboflow import Roboflow
import supervision as sv
from PIL import Image
import tempfile
import os

# Roboflow model ID and API key
MODEL_ID = "fishing-net-hole-rullc/2"
API_KEY = "Q9X3pzJWmj3VBuDCp0kJ"

# Load model with Roboflow
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project, version = MODEL_ID.split("/")
    return rf.workspace().project(project).version(int(version)).model

model = load_model()

# Streamlit UI
st.title("🎣 Fishing Net Hole Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image_pil = Image.open(uploaded_file).convert("RGB")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_pil.save(tmp_file.name)
        temp_path = tmp_file.name

    # Run inference
    results = model.predict(temp_path).json()
    detections = sv.Detections.from_inference(results)

    # Confidence filtering
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1)
    detections = detections[detections.confidence > confidence_threshold]

    # Convert to OpenCV BGR format for drawing
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Annotate
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Display
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Net Holes", use_container_width=True)

    # Clean up temporary file
    os.remove(temp_path)
