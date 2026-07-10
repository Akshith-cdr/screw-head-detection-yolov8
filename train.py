from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a Roboflow screw-head dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/roboflow_screw_heads/data.yaml",
        help="Path to the dataset data.yaml file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO weights to start from.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Use -1 for auto batch sizing.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on, for example cpu, 0, 0,1.")
    parser.add_argument("--project", type=str, default="runs/train", help="Directory for training outputs.")
    parser.add_argument("--name", type=str, default="screw_head_yolov8n", help="Run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run name.")
    return parser


def resolve_data_path(data_path: str) -> Path:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset config at: {path}. Place the Roboflow export under data/roboflow_screw_heads/ so data.yaml exists."
        )
    return path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training with these settings:")
    print(f"  data    : {data_path}")
    print(f"  model   : {args.model}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  device  : {args.device if args.device is not None else 'default'}")
    print(f"  project : {project_dir}")
    print(f"  name    : {args.name}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=args.exist_ok,
        pretrained=True,
        verbose=True,
        save=True,
        save_period=-1,
        plots=True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir: Optional[Path] = None
    if hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)
    elif hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None) is not None:
        save_dir = Path(model.trainer.save_dir)

    if save_dir is not None:
        print()
        print("Training complete.")
        print(f"Run directory: {save_dir}")
        print(f"Best weights : {save_dir / 'weights' / 'best.pt'}")
        print(f"Last weights : {save_dir / 'weights' / 'last.pt'}")
    else:
        print()
        print("Training complete.")
        print("Ultralytics did not return a save directory, but the run finished successfully.")


if __name__ == "__main__":
    main()
