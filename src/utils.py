"""Shared helpers for the screw-head detection workflow."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

import yaml

try:
    from config import CLASS_COUNT, CLASS_NAMES, IMAGES_DIR_NAME, LABELS_DIR_NAME
except ImportError:  # Supports ``python -m src.<module>``.
    from src.config import CLASS_COUNT, CLASS_NAMES, IMAGES_DIR_NAME, LABELS_DIR_NAME


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a consistent command-line logger once per process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_directory(path: Path, description: str) -> Path:
    """Return an existing directory or raise a clear ``FileNotFoundError``."""
    if not path.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {path}")
    return path


def require_file(path: Path, description: str) -> Path:
    """Return an existing file or raise a clear ``FileNotFoundError``."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    return path


def ensure_directory(path: Path) -> Path:
    """Create a directory and its parents when necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path) -> list[Path]:
    """Return the directory's files sorted by filename, as in the notebook."""
    require_directory(directory, "Input")
    files = (item for item in directory.iterdir() if item.is_file())
    return sorted(files, key=lambda item: item.name)


def copy_files(files: Iterable[Path], destination: Path) -> int:
    """Copy files into ``destination`` and return the number copied."""
    ensure_directory(destination)
    count = 0
    for source in files:
        shutil.copy2(source, destination / source.name)
        count += 1
    return count


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    """Return the YOLO label path corresponding to an image filename."""
    label_path = labels_dir / f"{image_path.stem}.txt"
    return require_file(label_path, f"Label for image '{image_path.name}'")


def write_run_data_yaml(run_dir: Path) -> Path:
    """Write the per-fold Ultralytics dataset configuration used by the notebook."""
    data_yaml_path = run_dir / "data.yaml"
    payload = {
        "path": str(run_dir.resolve()),
        "train": f"train/{IMAGES_DIR_NAME}",
        "val": f"val/{IMAGES_DIR_NAME}",
        "nc": CLASS_COUNT,
        "names": CLASS_NAMES,
    }
    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return data_yaml_path


def dataset_subdirectories(dataset_dir: Path) -> tuple[Path, Path]:
    """Return the images and labels directories inside a dataset directory."""
    return dataset_dir / IMAGES_DIR_NAME, dataset_dir / LABELS_DIR_NAME
