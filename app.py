from __future__ import annotations

from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from ultralytics import YOLO


MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
SUPPORTED_EXTENSIONS = [
    "jpg", "jpeg", "png", "bmp", "dib", "tif", "tiff", "webp", "jfif",
    "heic", "heif",
]
SUPPORTED_FORMATS = {
    "JPEG", "PNG", "BMP", "DIB", "TIFF", "WEBP", "HEIF", "HEIC",
}


def configure_heif_support() -> None:
    """Register Pillow's HEIC/HEIF decoder once for the current process."""
    register_heif_opener()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """Load and cache the trained YOLO model for all app reruns."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError("The trained model was not found at models/best.pt.")
    return YOLO(path)


def load_image(uploaded_file: Any) -> Image.Image:
    """Validate an uploaded image and return it as a loaded RGB PIL image."""
    try:
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise ValueError("The selected file is empty.")

        with Image.open(BytesIO(image_bytes)) as opened_image:
            image_format = (opened_image.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {image_format or 'unknown'}."
                )
            opened_image.load()
            return opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(
            "This file is not a supported, readable image. Please choose a valid "
            "JPG, PNG, BMP, TIFF, WebP, or HEIC/HEIF image."
        ) from error


def run_inference(
    model: YOLO, image: Image.Image, confidence: float
) -> tuple[Any, Image.Image, float]:
    """Run YOLO inference and return the result, annotated image, and elapsed time."""
    started_at = perf_counter()
    results = model.predict(
        source=image,
        conf=confidence,
        agnostic_nms=True,
        verbose=False,
    )
    elapsed_seconds = perf_counter() - started_at

    if not results:
        raise RuntimeError("The model returned no inference result.")

    # Ultralytics returns a BGR NumPy array from plot(); reverse channels for PIL.
    annotated_bgr = results[0].plot()
    annotated_image = Image.fromarray(annotated_bgr[:, :, ::-1])
    return results[0], annotated_image, elapsed_seconds


def image_to_png(image: Image.Image) -> bytes:
    """Encode an image as PNG bytes for Streamlit's download button."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_sidebar() -> float:
    """Render sidebar content and return the selected confidence threshold."""
    with st.sidebar:
        st.header("Model Information")
        st.write("**Model:** YOLOv8n")
        st.write("**Framework:** Ultralytics")
        st.write("**Cross Validation:** 5-fold")
        st.write(
            "**Supported Formats:** JPG, JPEG, PNG, BMP, DIB, TIFF, WebP, "
            "JFIF, HEIC, HEIF"
        )
        st.divider()
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.20,
            step=0.01,
            help="Only detections at or above this confidence are displayed.",
        )
        st.divider()
        st.header("About")
        st.write(
            "Upload or capture an image to locate screw heads using the project's "
            "trained YOLOv8 model. Images are processed in memory only."
        )
    return confidence


def render_results(
    result: Any,
    image: Image.Image,
    annotated: Image.Image,
    inference_time: float,
) -> None:
    """Render input, annotations, summary metrics, and downloadable result."""
    st.divider()
    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
    with right_column:
        st.subheader("Detection Result")
        st.image(annotated, use_container_width=True)

    st.divider()
    st.subheader("Detection Summary")
    boxes = result.boxes
    detection_count = 0 if boxes is None else len(boxes)

    if detection_count == 0:
        st.success("Detection completed successfully.")
        st.info(
            "No screw heads were detected. Possible reasons:\n\n"
            "- The screw head occupies only a small area.\n"
            "- The image differs from the training data.\n"
            "- The confidence threshold is too high.\n\n"
            "Try lowering the confidence threshold and run detection again."
        )
    else:
        rows: list[dict[str, str]] = []
        confidences: list[float] = []
        coordinates = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        for number, (coordinate, score) in enumerate(zip(coordinates, scores), 1):
            confidence_percent = float(score) * 100
            confidences.append(confidence_percent)
            x1, y1, x2, y2 = coordinate
            rows.append({
                "Detection Number": str(number),
                "Confidence (%)": f"{confidence_percent:.2f}",
                "Bounding Box Coordinates": (
                    f"({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})"
                ),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Detected Objects", detection_count)
        metric_two.metric("Average Confidence", f"{np.mean(confidences):.2f}%")
        metric_three.metric("Inference Time", f"{inference_time:.2f} s")

    try:
        st.download_button(
            "Download Detection Result",
            data=image_to_png(annotated),
            file_name="screw_head_detection.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:
        st.error("The detection result could not be prepared for download.")


def main() -> None:
    """Configure and run the Streamlit application."""
    st.set_page_config(page_title="Screw Head Detection", layout="centered")
    configure_heif_support()
    st.markdown(
        """<style>
        @media (max-width: 640px) {
          [data-testid="stHorizontalBlock"] { flex-direction: column; }
        }
        </style>""",
        unsafe_allow_html=True,
    )

    confidence = render_sidebar()
    st.title("Screw Head Detection using YOLOv8")
    st.write(
        "Upload an image or take a photo to detect screw heads with the trained "
        "YOLOv8 model."
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=SUPPORTED_EXTENSIONS,
        help="Choose a supported image file from your device.",
    )
    camera_file = st.camera_input("Take Photo")
    selected_file = uploaded_file if uploaded_file is not None else camera_file

    if selected_file is None:
        st.caption("An uploaded image takes priority when both options are used.")
        return

    try:
        image = load_image(selected_file)
    except Exception as error:
        st.error(f"Unable to open the selected image. {error}")
        return

    if st.button("Run Detection", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading model..."):
                model = load_model(str(MODEL_PATH))
        except FileNotFoundError as error:
            st.error(f"Model loading failed. {error}")
            return
        except Exception:
            st.error("Model loading failed. Please verify the deployment configuration.")
            return

        try:
            with st.spinner("Running detection..."):
                result, annotated, elapsed = run_inference(model, image, confidence)
            render_results(result, image, annotated, elapsed)
        except Exception:
            st.error(
                "Detection could not be completed. Please try another image or "
                "try again shortly."
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Keep unexpected Streamlit rendering errors from exposing implementation details.
        st.error(
            "The application encountered an unexpected error. Please refresh and "
            "try again."
        )
