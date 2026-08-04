"""
Converts labelme JSON annotations into YOLO-pose format (for YOLOv8-pose),
one dataset per view family. Bounding box is computed PER-IMAGE from that
image's own landmark extents + padding. For views where landmarks are
nearly collinear (basal: horizontal line; superior: near-vertical line),
the ill-defined dimension is NOT derived from landmark spread (which would
be near-zero and meaningless) — instead it's set equal to the padded
extent of the well-defined dimension, approximating a square region.
"""
import numpy as np
from pathlib import Path
from PIL import Image

from ..datasets.labelme_parser import parse_labelme_json
from ..datasets.splits import get_split_json_paths
from ..taxonomy import VIEW_LANDMARKS, FOLDER_TO_VIEW_FAMILY, landmarks_for

DATA_ROOT = Path("/content/drive/MyDrive/Rhinoplasty_Landmark/views-02")
SPLIT_PATH = Path("/content/drive/MyDrive/Rhinoplasty_Landmark/patient_split.csv")
OUT_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-data")

PADDING = 0.3         
MIN_BOX_FRAC = 0.12      # absolute floor: box never smaller than this fraction of image dim

# Views where one landmark axis is near-collinear and shouldn't be trusted directly.
# "y" = height is degenerate (landmarks spread horizontally, e.g. basal)
# "x" = width is degenerate (landmarks spread vertically, e.g. superior)
DEGENERATE_AXIS = {
    "basal": "y",
    "superior": "x",
}

VIEW_TO_FOLDERS = {}
for folder, (vf, _) in FOLDER_TO_VIEW_FAMILY.items():
    VIEW_TO_FOLDERS.setdefault(vf, []).append(folder)


def compute_box_for_image(parsed, view_family):
    pts = np.array(list(parsed["points"].values()))
    iw, ih = parsed["image_width"], parsed["image_height"]

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx_px, cy_px = (x_min + x_max) / 2, (y_min + y_max) / 2

    raw_w, raw_h = x_max - x_min, y_max - y_min
    pad_w = raw_w * (1 + 2 * PADDING)
    pad_h = raw_h * (1 + 2 * PADDING)

    axis = DEGENERATE_AXIS.get(view_family)
    if axis == "y":
        pad_h = pad_w   # approximate as square, using the reliable width
    elif axis == "x":
        pad_w = pad_h   # approximate as square, using the reliable height

    # absolute safety floor, regardless of view
    pad_w = max(pad_w, MIN_BOX_FRAC * iw)
    pad_h = max(pad_h, MIN_BOX_FRAC * ih)

    x_min_box = max(0, cx_px - pad_w / 2)
    x_max_box = min(iw, cx_px + pad_w / 2)
    y_min_box = max(0, cy_px - pad_h / 2)
    y_max_box = min(ih, cy_px + pad_h / 2)

    cx = (x_min_box + x_max_box) / 2 / iw
    cy = (y_min_box + y_max_box) / 2 / ih
    w = (x_max_box - x_min_box) / iw
    h = (y_max_box - y_min_box) / ih
    return cx, cy, w, h


def convert_view(view_family):
    landmark_order = landmarks_for(view_family)

    for split in ["train", "val", "test"]:
        img_out_dir = OUT_ROOT / view_family / "images" / split
        lbl_out_dir = OUT_ROOT / view_family / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for folder in VIEW_TO_FOLDERS[view_family]:
            json_paths = get_split_json_paths(DATA_ROOT / folder, split, SPLIT_PATH)
            for jp in json_paths:
                parsed = parse_labelme_json(jp)
                if len(parsed["points"]) == 0:
                    continue

                img_path = jp.with_suffix(".jpg")
                if not img_path.exists():
                    for ext in [".jpeg", ".JPG", ".png"]:
                        alt = jp.with_suffix(ext)
                        if alt.exists():
                            img_path = alt
                            break

                img = Image.open(img_path).convert("RGB")
                if parsed["flipped"]:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                out_stem = jp.stem
                img.save(img_out_dir / f"{out_stem}.jpg", quality=95)

                cx, cy, w, h = compute_box_for_image(parsed, view_family)

                iw, ih = parsed["image_width"], parsed["image_height"]
                kpt_parts = []
                for name in landmark_order:
                    if name in parsed["points"]:
                        x, y = parsed["points"][name]
                        kpt_parts += [f"{x/iw:.6f}", f"{y/ih:.6f}", "2"]
                    else:
                        kpt_parts += ["0.0", "0.0", "0"]

                line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kpt_parts)
                with open(lbl_out_dir / f"{out_stem}.txt", "w") as f:
                    f.write(line + "\n")

        n_files = len(list(img_out_dir.glob("*.jpg")))
        print(f"{view_family} {split}: {n_files} images")

    num_kpts = len(landmark_order)
    yaml_content = f"""path: {OUT_ROOT / view_family}
train: images/train
val: images/val
test: images/test

kpt_shape: [{num_kpts}, 3]
names:
  0: landmark_region
"""
    with open(OUT_ROOT / view_family / "dataset.yaml", "w") as f:
        f.write(yaml_content)


def main():
    for view_family in VIEW_LANDMARKS.keys():
        convert_view(view_family)
        print()


if __name__ == "__main__":
    main()