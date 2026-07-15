from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

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
    "Upload an image to detect screw heads using the trained YOLOv8 model."
)

st.divider()

# Sidebar

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** YOLOv8n")
    st.write("**Framework:** Ultralytics")
    st.write("**Classes:** 1")
    st.write("**Image Size:** 640 × 640")
    st.write("**Cross Validation:** 5-Fold")
    st.write("**Confidence Threshold:** 0.25")

# Image Upload

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.success("Image uploaded successfully.")

    if st.button("Run Detection", type="primary", use_container_width=True):

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
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

        st.divider()

        st.subheader("Detection Summary")

        boxes = result.boxes

        if len(boxes) == 0:

            st.warning("No screw heads detected.")

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

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Detected Screw Heads",
                    len(boxes)
                )

            with c2:
                st.metric(
                    "Average Confidence",
                    f"{np.mean(confidences):.2f}%"
                )

st.divider()

st.caption(
    "Developed using YOLOv8 • Ultralytics • PyTorch • Streamlit"
)