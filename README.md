# Screw Head Detection with YOLOv8

Baseline training project for screw-head object detection using a Roboflow dataset and Ultralytics YOLOv8.

## What this project does

- Fine-tunes a pretrained `yolov8n.pt` baseline on your custom dataset.
- Validates the trained model on the dataset validation split.
- Runs prediction on test images or any image folder.
- Saves outputs in a clean and repeatable folder structure.

## Folder layout

```text
Screw_Head_Detection_YOLOv8/
  data/
    README.md
    roboflow_screw_heads/
      data.yaml
      train/
      valid/
      test/
  train.py
  validate.py
  predict.py
  requirements.txt
  .gitignore
  README.md
```

## Step 1: Create a virtual environment

If your Python install is unstable, use a virtual environment before anything else.

On Windows CMD:

```bat
cd c:\Users\veeri\Documents\Projects\Screw_Head_Detection_YOLOv8
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

If `py -3.11` is not available, use:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## Step 2: Install dependencies

```bat
pip install -r requirements.txt
```

## Step 3: Download the Roboflow dataset

1. Open your Roboflow project.
2. Click export/download.
3. Choose **YOLO** format, which produces `data.yaml`, `train/`, `valid/`, and `test/`.
4. Download the export as a zip.
5. Extract it into:

```text
data/roboflow_screw_heads/
```

After extraction, this file must exist:

```text
data/roboflow_screw_heads/data.yaml
```

## Step 4: Train the baseline model

```bat
python train.py --data data/roboflow_screw_heads/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --name screw_head_yolov8n
```

Expected output:

- training logs in the terminal
- weights saved under `runs/detect/runs/train/screw_head_yolov8n/weights/`
- best model at `runs/detect/runs/train/screw_head_yolov8n/weights/best.pt`

## Step 5: Validate the model

```bat
python validate.py --data data/roboflow_screw_heads/data.yaml
```

Expected output:

- validation metrics in the terminal
- optional plots saved under `runs/val/screw_head_yolov8n_val/`

## Step 6: Run prediction on test images

You can point prediction at the Roboflow test folder:

```bat
python predict.py --source data/roboflow_screw_heads/test/images --imgsz 640 --conf 0.25 --save-txt
```

Expected output:

- annotated images saved under `outputs/predict/screw_head_yolov8n_preds/`
- optional label text files if `--save-txt` is used

## Common mistakes and fixes

- Wrong `data.yaml` path: make sure the file is exactly at `data/roboflow_screw_heads/data.yaml`.
- Missing Ultralytics install: run `pip install -r requirements.txt` inside the virtual environment.
- CUDA not detected: use `--device cpu` for a CPU-only run, or install the correct GPU-enabled PyTorch stack on the GPU machine.
- Empty prediction output: check that `--source` points to real images, not the parent dataset folder.
- Training starts but cannot find images: confirm the Roboflow export was extracted fully, including `train/`, `valid/`, and `test/`.

## Next iteration ideas

- Increase image size to `768` or `1024` if the screw heads are very small.
- Try a larger model such as `yolov8s.pt` after the baseline is working.
- Increase epochs to `100` once the pipeline is stable.
- Tune `batch`, `conf`, and augmentation settings after reviewing validation results.
