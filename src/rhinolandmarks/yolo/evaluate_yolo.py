"""
Runs a trained YOLOv8-pose model on its view family's test set, and scores
predictions using the SAME NME/PCK code (utils/metrics.py) used for the
U-Net models — this is what makes the two paradigms directly comparable.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from ultralytics import YOLO

from rhinolandmarks.datasets.labelme_parser import parse_labelme_json
from rhinolandmarks.datasets.splits import get_split_json_paths
from rhinolandmarks.utils.metrics import compute_nme
from rhinolandmarks.taxonomy import landmarks_for, NORM_LANDMARK_PAIRS, FOLDER_TO_VIEW_FAMILY

DATA_ROOT = Path("/content/drive/MyDrive/Rhinoplasty_Landmark/views-02")   # original JSONs, for ground truth
SPLIT_PATH = Path("/content/drive/MyDrive/Rhinoplasty_Landmark/patient_split.csv")
YOLO_RUNS_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-runs")
YOLO_DATA_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/yolo-data")

PCK_THRESHOLDS = [0.05, 0.10, 0.20]

VIEW_TO_FOLDERS = {}
for folder, (vf, _) in FOLDER_TO_VIEW_FAMILY.items():
    VIEW_TO_FOLDERS.setdefault(vf, []).append(folder)


def evaluate_view(view_family):
    landmark_order = landmarks_for(view_family)
    norm_pair = NORM_LANDMARK_PAIRS[view_family]

    weights_path = YOLO_RUNS_ROOT / view_family / "weights" / "best.pt"
    model = YOLO(str(weights_path))

    test_img_dir = YOLO_DATA_ROOT / view_family / "images" / "test"
    all_errors = []
    image_nmes = []

    for img_path in sorted(test_img_dir.glob("*.jpg")):
        stem = img_path.stem

        # Ground truth: re-parse the ORIGINAL labelme json (same source used
        # to build the U-Net's targets), so both paradigms are scored
        # against identically-defined ground truth.
        orig_json_path = None
        for folder in VIEW_TO_FOLDERS[view_family]:
            candidate = DATA_ROOT / folder / f"{stem}.json"
            if candidate.exists():
                orig_json_path = candidate
                break
        if orig_json_path is None:
            print(f"WARNING: no matching original json for {stem}, skipping")
            continue

        parsed = parse_labelme_json(orig_json_path)
        gt_coords = np.zeros((len(landmark_order), 2), dtype=np.float32)
        visible = np.zeros(len(landmark_order), dtype=np.float32)
        for c, name in enumerate(landmark_order):
            if name in parsed["points"]:
                gt_coords[c] = parsed["points"][name]
                visible[c] = 1.0

        # Prediction
        result = model.predict(source=str(img_path), verbose=False)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue  # no detection at all for this image

        # Highest-confidence detection (should normally be the only one, single-instance-per-image design)
        if result.boxes is not None and len(result.boxes.conf) > 0:
            best_idx = int(result.boxes.conf.argmax())
        else:
            best_idx = 0

        pred_coords = result.keypoints.xy[best_idx].cpu().numpy()  # (num_kpts, 2), already in ORIGINAL image pixel space

        nme = compute_nme(pred_coords, gt_coords, visible, landmark_order, norm_pair)
        if nme is not None:
            image_nmes.append(nme)

        per_landmark_error = np.linalg.norm(pred_coords - gt_coords, axis=1)
        idx_a = landmark_order.index(norm_pair[0])
        idx_b = landmark_order.index(norm_pair[1])
        if visible[idx_a] and visible[idx_b]:
            norm_dist = np.linalg.norm(gt_coords[idx_a] - gt_coords[idx_b])
            if norm_dist > 1e-6:
                for c, name in enumerate(landmark_order):
                    if visible[c] == 1:
                        all_errors.append((stem, name, per_landmark_error[c] / norm_dist))

    err_df = pd.DataFrame(all_errors, columns=["image", "landmark", "normalized_error"])

    landmark_stats = err_df.groupby("landmark")["normalized_error"].agg(
        mean="mean", median="median", std="std", n="count"
    ).reset_index()
    for t in PCK_THRESHOLDS:
        pck = err_df.groupby("landmark")["normalized_error"].apply(lambda x: (x <= t).mean())
        landmark_stats[f"pck@{t}"] = landmark_stats["landmark"].map(pck)
    landmark_stats.insert(0, "view_family", view_family)
    landmark_stats.insert(0, "model", "yolov8n-pose")

    view_row = {
        "model": "yolov8n-pose",
        "view_family": view_family,
        "n_evaluated_images": len(image_nmes),
        "mean_NME": np.mean(image_nmes) if image_nmes else None,
        "median_NME": np.median(image_nmes) if image_nmes else None,
        "std_NME": np.std(image_nmes) if image_nmes else None,
    }
    for t in PCK_THRESHOLDS:
        view_row[f"pck@{t}"] = (err_df["normalized_error"] <= t).mean() if len(err_df) else None

    return landmark_stats, view_row, err_df


def main():
    all_landmark_stats, all_view_rows, all_raw = [], [], []
    for view_family in ["frontal", "basal", "lateral", "oblique", "superior"]:
        weights_path = YOLO_RUNS_ROOT / view_family / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"Skipping {view_family} — no trained weights found yet.")
            continue
        landmark_stats, view_row, err_df = evaluate_view(view_family)
        all_landmark_stats.append(landmark_stats)
        all_view_rows.append(view_row)
        err_df["view_family"] = view_family
        all_raw.append(err_df)

    out_dir = Path("/content/drive/MyDrive/rhino-landmarks-data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_landmark_stats, ignore_index=True).to_csv(out_dir / "yolo_test_results_per_landmark.csv", index=False)
    pd.DataFrame(all_view_rows).to_csv(out_dir / "yolo_test_results_per_view.csv", index=False)
    pd.concat(all_raw, ignore_index=True).to_csv(out_dir / "yolo_test_results_raw_errors.csv", index=False)

    print(pd.DataFrame(all_view_rows).to_string(index=False))


if __name__ == "__main__":
    main()