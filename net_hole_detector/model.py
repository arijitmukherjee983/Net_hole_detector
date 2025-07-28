import numpy as np
import cv2
import supervision as sv

def predict(model, image_pil, temp_path, confidence_threshold=0.1):
    results = model.predict(temp_path).json()
    detections = sv.Detections.from_inference(results)
    detections = detections[detections.confidence > confidence_threshold]

    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    # Convert to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return detections, annotated_rgb
