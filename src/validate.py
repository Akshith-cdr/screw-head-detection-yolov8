"""Summarize final YOLO metrics from all five cross-validation runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from config import CROSS_VALIDATION_RESULTS_PATH, FOLD_COUNT, METRIC_COLUMNS, RESULTS_DIR
    from utils import configure_logging, require_file
except ImportError:
    from src.config import CROSS_VALIDATION_RESULTS_PATH, FOLD_COUNT, METRIC_COLUMNS, RESULTS_DIR
    from src.utils import configure_logging, require_file


LOGGER = logging.getLogger(__name__)


def collect_cross_validation_results(results_dir: Path, output_path: Path) -> pd.DataFrame:
    """Read each final training row and save notebook-equivalent aggregate metrics."""
    summary_rows: list[dict[str, float | int]] = []
    for fold_number in range(1, FOLD_COUNT + 1):
        csv_path = require_file(
            results_dir / f"fold_{fold_number}" / "results.csv",
            f"Results for fold {fold_number}",
        )
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError(f"Results file is empty: {csv_path}")
        final_row = frame.iloc[-1]
        missing = [column for column in METRIC_COLUMNS.values() if column not in frame.columns]
        if missing:
            raise ValueError(f"Results file {csv_path} is missing columns: {', '.join(missing)}")
        metrics = {
            name: float(final_row[column])
            for name, column in METRIC_COLUMNS.items()
        }
        summary_rows.append({"Fold": fold_number, **metrics})

    summary = pd.DataFrame(summary_rows)
    metrics = list(METRIC_COLUMNS)
    average = summary[metrics].mean()
    standard_deviation = summary[metrics].std()
    # Keep the exported layout used by the notebook: aggregate rows have "-"
    # in Fold, while their order is Average followed by Std Dev.
    summary.loc[len(summary)] = {"Fold": "-", **average.to_dict()}
    summary.loc[len(summary)] = {"Fold": "-", **standard_deviation.to_dict()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    LOGGER.info("Saved cross-validation results to %s.", output_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=CROSS_VALIDATION_RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    """Generate ``cross_validation_results.csv``."""
    configure_logging()
    args = parse_args()
    try:
        collect_cross_validation_results(args.results_dir, args.output)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.error("Cross-validation summary failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
