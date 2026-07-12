"""Run YOLOv8 inference on one image."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

try:
    from config import DEFAULT_PREDICTION_MODEL, IMAGE_SIZE, PREDICTION_RESULTS_DIR
    from utils import configure_logging, require_file
except ImportError:
    from src.config import DEFAULT_PREDICTION_MODEL, IMAGE_SIZE, PREDICTION_RESULTS_DIR
    from src.utils import configure_logging, require_file


LOGGER = logging.getLogger(__name__)


def predict(image_path: Path, model_path: Path, output_dir: Path, image_size: int) -> None:
    """Load a selected YOLO model and save inference output for one image."""
    require_file(image_path, "Input image")
    require_file(model_path, "Model")
    model = YOLO(str(model_path))
    model.predict(
        source=str(image_path),
        imgsz=image_size,
        project=str(output_dir),
        name="predict",
        exist_ok=True,
        save=True,
    )
    LOGGER.info("Prediction saved under %s.", output_dir / "predict")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Image to analyse.")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_PREDICTION_MODEL,
        help="YOLO .pt model path.",
    )
    parser.add_argument("--output-dir", type=Path, default=PREDICTION_RESULTS_DIR)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


def main() -> None:
    """Run single-image inference."""
    configure_logging()
    args = parse_args()
    try:
        predict(args.image, args.model, args.output_dir, args.imgsz)
    except (FileNotFoundError, OSError, ValueError) as error:
        LOGGER.error("Prediction failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
