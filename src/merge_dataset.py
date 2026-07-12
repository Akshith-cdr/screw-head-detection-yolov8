"""Merge the Roboflow train and validation splits into one dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from config import (COMBINED_DATASET_DIR, DATASET_DIR, IMAGES_DIR_NAME,
                        LABELS_DIR_NAME, TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME)
    from utils import (configure_logging, copy_files, dataset_subdirectories,
                       ensure_directory, list_files, require_directory)
except ImportError:
    from src.config import (COMBINED_DATASET_DIR, DATASET_DIR, IMAGES_DIR_NAME,
                            LABELS_DIR_NAME, TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME)
    from src.utils import (configure_logging, copy_files, dataset_subdirectories,
                           ensure_directory, list_files, require_directory)


LOGGER = logging.getLogger(__name__)


def merge_dataset(dataset_dir: Path, output_dir: Path) -> None:
    """Copy the original train and valid files into a combined dataset directory."""
    require_directory(dataset_dir, "Dataset")
    output_images, output_labels = dataset_subdirectories(output_dir)
    ensure_directory(output_images)
    ensure_directory(output_labels)

    for split_name in (TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME):
        split_images, split_labels = dataset_subdirectories(dataset_dir / split_name)
        image_count = copy_files(list_files(split_images), output_images)
        label_count = copy_files(list_files(split_labels), output_labels)
        LOGGER.info("Copied %d images and %d labels from %s.", image_count, label_count, split_name)

    LOGGER.info(
        "Combined dataset contains %d images and %d labels.",
        len(list_files(output_images)),
        len(list_files(output_labels)),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help="Roboflow dataset directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=COMBINED_DATASET_DIR,
        help="Combined dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset merge command."""
    configure_logging()
    args = parse_args()
    try:
        merge_dataset(args.dataset_dir, args.output_dir)
    except (FileNotFoundError, OSError) as error:
        LOGGER.error("Dataset merge failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
