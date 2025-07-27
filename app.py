import streamlit as st
import numpy as np
import cv2
from inference import get_model
import supervision as sv
from PIL import Image

# Roboflow model ID and API key
MODEL_ID = "fishing-net-hole-rullc/2"
API_KEY = "Q9X3pzJWmj3VBuDCp0kJ"

# Load the Roboflow model (cached for performance)
@st.cache_resource
def load_model():
    return get_model(model_id=MODEL_ID, api_key=API_KEY)

model = load_model()

# Streamlit UI
st.title("🎣 Fishing Net Hole Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and convert uploaded file to OpenCV format
    image_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Confidence filtering
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1)
    detections = detections[detections.confidence > confidence_threshold]

    # Annotate
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Convert BGR to RGB for Streamlit display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Detected Net Holes", use_container_width=True)

