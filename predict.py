from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 predictions on images or folders.")
    parser.add_argument("--weights", type=str, default=None, help="Path to the trained weights file.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video, or webcam source. Use a folder of test images for quick evaluation.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, for example cpu or 0.")
    parser.add_argument("--project", type=str, default="outputs/predict", help="Directory for prediction outputs.")
    parser.add_argument("--name", type=str, default="screw_head_yolov8n_preds", help="Run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as text files.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence values with text labels.")
    parser.add_argument("--show", action="store_true", help="Display prediction windows if supported.")
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
        "Weights file not found. Pass --weights explicitly or make sure training finished and saved best.pt under runs/."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    weights_path = resolve_weights_path(args.weights)
    source_path = Path(args.source)

    if not source_path.exists() and not str(args.source).startswith(("0", "1", "2", "3")):
        raise FileNotFoundError(f"Source not found: {source_path}")

    print("Running prediction with these settings:")
    print(f"  weights : {weights_path}")
    print(f"  source  : {args.source}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  conf    : {args.conf}")
    print(f"  iou     : {args.iou}")
    print(f"  device  : {args.device if args.device is not None else 'default'}")

    model = YOLO(str(weights_path))

    predict_kwargs = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        verbose=True,
    )

    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    print()
    print("Prediction complete.")
    print(f"Saved results to: {Path(args.project) / args.name}")
    print(f"Number of processed items: {len(results)}")


if __name__ == "__main__":
    main()
