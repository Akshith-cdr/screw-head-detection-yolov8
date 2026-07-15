from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
from ultralytics import YOLO

# Enable HEIC/HEIF support
register_heif_opener()

# Page Configuration

st.set_page_config(
    page_title="Screw Head Detection",
    layout="centered"
)

# Load Model

MODEL_PATH = Path("models") / "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# Header

st.title("Screw Head Detection using YOLOv8")

st.write(
    """
Upload an image to detect screw heads using the trained YOLOv8 object detection model.
Supported image formats: JPG, JPEG, PNG, HEIC and HEIF.
"""
)

st.divider()

# Sidebar

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** YOLOv8n")
    st.write("**Framework:** Ultralytics")
    st.write("**Dataset:** Roboflow")
    st.write("**Classes:** 1")
    st.write("**Image Size:** 640 × 640")
    st.write("**Cross Validation:** 5-Fold")
    st.write("**Confidence Threshold:** 0.25")

# Image Upload

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "heic", "heif"],
    help="Supported formats: JPG, JPEG, PNG, HEIC and HEIF"
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:
        st.error(f"Unable to open the uploaded image.\n\n{e}")
        st.stop()

    st.success("Image uploaded successfully.")

    if st.button("Run Detection", use_container_width=True):

        with st.spinner("Running inference..."):

            results = model.predict(
                source=image,
                conf=0.25,
                verbose=False
            )

        result = results[0]

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, width=400)

        with col2:
            st.subheader("Detection Result")
            st.image(annotated, width=400)

        st.divider()

        st.subheader("Detection Summary")

        boxes = result.boxes

        if len(boxes) == 0:

            st.warning("No screw heads were detected.")

        else:

            confidences = []

            table = []

            for i, box in enumerate(boxes, start=1):

                conf = float(box.conf[0]) * 100
                confidences.append(conf)

                table.append({
                    "Detection": f"Screw Head {i}",
                    "Confidence (%)": f"{conf:.2f}"
                })

            st.table(table)

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