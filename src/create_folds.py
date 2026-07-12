"""Create five deterministic validation folds from the merged dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import KFold

try:
    from config import COMBINED_DATASET_DIR, FOLD_COUNT, FOLDS_DIR, RANDOM_STATE
    from utils import (configure_logging, copy_files, dataset_subdirectories,
                       ensure_directory, label_path_for_image, list_files,
                       require_directory)
except ImportError:
    from src.config import COMBINED_DATASET_DIR, FOLD_COUNT, FOLDS_DIR, RANDOM_STATE
    from src.utils import (configure_logging, copy_files, dataset_subdirectories,
                           ensure_directory, label_path_for_image, list_files,
                           require_directory)


LOGGER = logging.getLogger(__name__)


def create_folds(combined_dir: Path, folds_dir: Path) -> None:
    """Create notebook-equivalent KFold validation directories with their labels."""
    image_dir, label_dir = dataset_subdirectories(combined_dir)
    require_directory(image_dir, "Combined images")
    require_directory(label_dir, "Combined labels")
    images = list_files(image_dir)
    if len(images) < FOLD_COUNT:
        raise ValueError(f"At least {FOLD_COUNT} images are required; found {len(images)}.")

    kfold = KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
    for fold_number, (_, validation_indices) in enumerate(kfold.split(images), start=1):
        fold_dir = ensure_directory(folds_dir / f"fold{fold_number}")
        fold_images, fold_labels = dataset_subdirectories(fold_dir)
        validation_images = [images[index] for index in validation_indices]
        copy_files(validation_images, fold_images)
        labels = (
            label_path_for_image(image, label_dir)
            for image in validation_images
        )
        copy_files(labels, fold_labels)
        LOGGER.info("Fold %d contains %d images and labels.", fold_number, len(validation_images))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, default=COMBINED_DATASET_DIR)
    parser.add_argument("--folds-dir", type=Path, default=FOLDS_DIR)
    return parser.parse_args()


def main() -> None:
    """Run fold creation."""
    configure_logging()
    args = parse_args()
    try:
        create_folds(args.combined_dir, args.folds_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        LOGGER.error("Fold creation failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
