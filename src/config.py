"""Project-wide configuration for the cross-validation workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
COMBINED_DATASET_DIR = PROJECT_ROOT / "combined_dataset"
FOLDS_DIR = PROJECT_ROOT / "folds"
CV_RUNS_DIR = PROJECT_ROOT / "cv_runs"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CROSS_VALIDATION_RESULTS_PATH = PROJECT_ROOT / "cross_validation_results.csv"

TRAIN_SPLIT_NAME = "train"
VALIDATION_SPLIT_NAME = "valid"
IMAGES_DIR_NAME = "images"
LABELS_DIR_NAME = "labels"

CLASS_NAMES = ["screw"]
CLASS_COUNT = len(CLASS_NAMES)
FOLD_COUNT = 5
RANDOM_STATE = 42
MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEFAULT_PREDICTION_MODEL = MODELS_DIR / "best.pt"
PREDICTION_RESULTS_DIR = RESULTS_DIR / "predictions"

METRIC_COLUMNS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
