"""
Trains YOLOv8n-pose from RANDOM INITIALIZATION (architecture only, no
pretrained weights), as the scratch-initialized counterpart to the
COCO-pretrained YOLOv8n-pose already trained in train_yolo.py.
Usage: python -m rhinolandmarks.yolo.train_yolo_scratch --view lateral
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

YOLO_DATA_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-data")
YOLO_RUNS_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-runs-scratch")

# Same epoch ceiling as the pretrained YOLO run for nominal parity, but
# note: per He et al. (2019), scratch-initialized models typically need
# MORE iterations to converge than pretrained ones. If training clearly
# hasn't plateaued by 150 epochs, this should be extended and reported
# as such (asymmetric budget, honestly justified) rather than silently
# cut off. Larger patience gives it more room within the same ceiling.
EPOCHS = 150
PATIENCE = 20
                        # to need longer to plateau; using the same tight patience as the
                        # pretrained run risks premature stopping before convergence
IMG_SIZE = 640
BATCH_SIZE = 16
ARCH_YAML = "yolov8n-pose.yaml"   # architecture definition only — NO pretrained weights loaded


def main(view):
    data_yaml = YOLO_DATA_ROOT / view / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found — run convert_to_yolo.py first.")

    model = YOLO(ARCH_YAML)   # random-initialized architecture, NOT YOLO("yolov8n-pose.pt")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(YOLO_RUNS_ROOT),
        name=view,
        exist_ok=True,
        seed=42,
        val=True,
        plots=True,
        pretrained=False,   # explicit, redundant with using the .yaml, but documents intent clearly
    )

    print(f"\nDone. Best weights at: {YOLO_RUNS_ROOT / view / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=str, required=True,
                         choices=["frontal", "basal", "lateral", "oblique", "superior"])
    args = parser.parse_args()
    main(args.view)