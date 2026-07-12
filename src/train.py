"""Build cross-validation datasets and train YOLOv8n once for each fold."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

try:
    from config import (BATCH_SIZE, CV_RUNS_DIR, EPOCHS, FOLD_COUNT, FOLDS_DIR,
                        IMAGE_SIZE, MODEL_NAME, RESULTS_DIR)
    from utils import (configure_logging, copy_files, dataset_subdirectories,
                       ensure_directory, list_files, require_directory,
                       write_run_data_yaml)
except ImportError:
    from src.config import (BATCH_SIZE, CV_RUNS_DIR, EPOCHS, FOLD_COUNT, FOLDS_DIR,
                            IMAGE_SIZE, MODEL_NAME, RESULTS_DIR)
    from src.utils import (configure_logging, copy_files, dataset_subdirectories,
                           ensure_directory, list_files, require_directory,
                           write_run_data_yaml)


LOGGER = logging.getLogger(__name__)


def build_run_dataset(validation_fold: int, folds_dir: Path, runs_dir: Path) -> Path:
    """Build one run dataset using one fold for validation and four for training."""
    run_dir = ensure_directory(runs_dir / f"run{validation_fold}")
    train_images, train_labels = dataset_subdirectories(ensure_directory(run_dir / "train"))
    validation_images, validation_labels = dataset_subdirectories(ensure_directory(run_dir / "val"))

    for fold_number in range(1, FOLD_COUNT + 1):
        source_images, source_labels = dataset_subdirectories(folds_dir / f"fold{fold_number}")
        require_directory(source_images, f"Fold {fold_number} images")
        require_directory(source_labels, f"Fold {fold_number} labels")
        if fold_number == validation_fold:
            target_images, target_labels = validation_images, validation_labels
        else:
            target_images, target_labels = train_images, train_labels
        copy_files(list_files(source_images), target_images)
        copy_files(list_files(source_labels), target_labels)

    data_yaml = write_run_data_yaml(run_dir)
    LOGGER.info(
        "Run %d prepared: %d train images, %d validation images.",
        validation_fold,
        len(list_files(train_images)),
        len(list_files(validation_images)),
    )
    return data_yaml


def train_folds(
    folds_dir: Path,
    runs_dir: Path,
    results_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    selected_fold: int | None = None,
) -> None:
    """Materialize datasets and train a new YOLO model for every requested fold."""
    folds = [selected_fold] if selected_fold is not None else list(range(1, FOLD_COUNT + 1))
    for fold_number in folds:
        if fold_number not in range(1, FOLD_COUNT + 1):
            raise ValueError(f"Fold must be between 1 and {FOLD_COUNT}.")
        data_yaml = build_run_dataset(fold_number, folds_dir, runs_dir)
        LOGGER.info("Training fold %d.", fold_number)
        model = YOLO(model_name)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=str(results_dir),
            name=f"fold_{fold_number}",
            exist_ok=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, default=FOLDS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=CV_RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--fold", type=int, choices=range(1, FOLD_COUNT + 1))
    return parser.parse_args()


def main() -> None:
    """Run cross-validation training."""
    configure_logging()
    args = parse_args()
    try:
        train_folds(
            args.folds_dir,
            args.runs_dir,
            args.results_dir,
            args.model,
            args.epochs,
            args.batch,
            args.imgsz,
            args.fold,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        LOGGER.error("Training failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
