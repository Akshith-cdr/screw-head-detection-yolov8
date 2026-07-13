import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration

st.set_page_config(
    page_title="Screw Head Detection",
    layout="wide"
)

# Load Model

MODEL_PATH = Path("models") / "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Header

st.title("Screw Head Detection using YOLOv8")

st.markdown(
    """
This application detects screw heads in an uploaded image using a trained YOLOv8 object detection model.

Upload an image and click **Run Detection** to perform inference.
"""
)

st.divider()

# Image Upload

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    if st.button("Run Detection", use_container_width=True):

        with st.spinner("Running inference..."):

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)

                results = model.predict(
                    source=tmp.name,
                    conf=0.25,
                    verbose=False
                )

        result = results[0]

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with right:
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

        st.divider()

        st.subheader("Detection Summary")

        boxes = result.boxes

        if len(boxes) == 0:

            st.info("No screw heads were detected in the uploaded image.")

        else:

            confidences = []

            for i, box in enumerate(boxes, start=1):

                confidence = float(box.conf[0]) * 100
                confidences.append(confidence)

                st.write(
                    f"Detection {i}: Screw Head ({confidence:.2f}% confidence)"
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Detected Screw Heads",
                    len(boxes)
                )

            with col2:
                st.metric(
                    "Average Confidence",
                    f"{np.mean(confidences):.2f}%"
                )

st.divider()

st.caption(
    "Screw Head Detection using YOLOv8 | Ultralytics • PyTorch • Streamlit"
)