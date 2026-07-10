from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained YOLOv8 model on the Roboflow dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/roboflow_screw_heads/data.yaml",
        help="Path to the dataset data.yaml file.",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to the trained weights file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to validate on, for example cpu or 0.")
    parser.add_argument("--project", type=str, default="runs/val", help="Directory for validation outputs.")
    parser.add_argument("--name", type=str, default="screw_head_yolov8n_val", help="Run name.")
    return parser


def resolve_weights_path(weights: Optional[str]) -> Path:
    if weights:
        path = Path(weights)
        if path.exists():
            return path

    candidates = sorted(
        Path("runs").rglob("best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Weights file not found. Pass --weights explicitly or make sure training finished and saved best.pt under runs/.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = Path(args.data)
    weights_path = resolve_weights_path(args.weights)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print("Running validation with these settings:")
    print(f"  data    : {data_path}")
    print(f"  weights : {weights_path}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  device  : {args.device if args.device is not None else 'default'}")

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True,
    )

    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)

    print()
    print("Validation complete.")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
