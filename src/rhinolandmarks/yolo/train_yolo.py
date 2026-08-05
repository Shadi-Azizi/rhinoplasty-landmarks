"""
Fine-tunes a pretrained YOLOv8-pose model on one view family's dataset.
Backbone weights transfer from COCO pretraining; the pose head is
automatically reinitialized to match this view's keypoint count
(kpt_shape in dataset.yaml), since it differs from COCO's 17 keypoints.

Usage: python -m rhinolandmarks.yolo.train_yolo --view lateral
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

YOLO_DATA_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-data")
YOLO_RUNS_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-runs")

# Same training discipline as the U-Net side: generous epoch ceiling,
# early stopping (Ultralytics calls this 'patience') decides actual length.
EPOCHS = 150
PATIENCE = 20
IMG_SIZE = 640          # YOLO's own input resizing, separate from our U-Net's 512x640
BATCH_SIZE = 16
PRETRAINED_WEIGHTS = "yolov8n-pose.pt"   # nano — fastest; swap to yolov8s-pose.pt if you want a larger model


def main(view):
    data_yaml = YOLO_DATA_ROOT / view / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found — run convert_to_yolo.py first.")

    model = YOLO(PRETRAINED_WEIGHTS)  # loads pretrained COCO-pose weights

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(YOLO_RUNS_ROOT),
        name=view,
        exist_ok=True,       # allows resuming into the same run folder
        seed=42,
        val=True,
        plots=True,           # Ultralytics auto-generates its own learning curve plots too
    )

    print(f"\nDone. Best weights at: {YOLO_RUNS_ROOT / view / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=str, required=True,
                         choices=["frontal", "basal", "lateral", "oblique", "superior"])
    args = parser.parse_args()
    main(args.view)