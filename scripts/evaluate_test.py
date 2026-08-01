import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from rhinolandmarks.models.registry import build_model
from rhinolandmarks.datasets.heatmap_dataset import ViewLandmarkDataset
from rhinolandmarks.datasets.splits import get_split_json_paths
from rhinolandmarks.utils.metrics import heatmaps_to_coords, compute_nme
from rhinolandmarks.taxonomy import landmarks_for, NORM_LANDMARK_PAIRS

# EDIT: one config file per view family, already trained
CONFIG_PATHS = [
    "configs/unet_frontal.yaml",
    "configs/unet_basal.yaml",
    "configs/unet_lateral.yaml",
    "configs/unet_oblique.yaml",
    "configs/unet_superior.yaml",
]

PCK_THRESHOLDS = [0.05, 0.10, 0.20]  # fraction of normalization distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pck(pred_coords, gt_coords, visible, landmark_order, norm_pair, thresholds):
    """
    Returns dict: threshold -> array of 0/1 per visible landmark (correct/not),
    or None per landmark if it wasn't visible. Same normalization logic as NME.
    """
    idx_a = landmark_order.index(norm_pair[0])
    idx_b = landmark_order.index(norm_pair[1])
    if visible[idx_a] == 0 or visible[idx_b] == 0:
        return None

    norm_dist = np.linalg.norm(gt_coords[idx_a] - gt_coords[idx_b])
    if norm_dist < 1e-6:
        return None

    per_landmark_error = np.linalg.norm(pred_coords - gt_coords, axis=1) / norm_dist
    return per_landmark_error  # per-landmark NORMALIZED error, used for PCK thresholds below


def evaluate_view(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    view_family = cfg["view_family"]
    landmark_order = landmarks_for(view_family)
    norm_pair = NORM_LANDMARK_PAIRS[view_family]
    num_channels = len(landmark_order)

    image_size = tuple(cfg["image_size"])
    heatmap_size = tuple(cfg["heatmap_size"])

    ckpt_path = Path(cfg["checkpoint_dir"]) / f"unet_{view_family}_best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model("unet", in_channels=3, out_channels=num_channels,
                         base_channels=cfg["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_root = Path(cfg["data_root"])
    split_path = Path(cfg["split_path"])
    test_json_paths = []
    for folder_name in cfg["view_folders"]:
        test_json_paths.extend(get_split_json_paths(data_root / folder_name, "test", split_path))

    test_ds = ViewLandmarkDataset(test_json_paths, image_size=image_size,
                                   heatmap_size=heatmap_size, sigma=cfg["heatmap_sigma"])

    print(f"\n{view_family}: {len(test_ds)} test images, checkpoint epoch {checkpoint['epoch']}, "
          f"val_NME={checkpoint['val_nme']:.4f}")

    # Collect per-image, per-landmark normalized error
    all_errors = []       # list of (image_idx, landmark_name, normalized_error)
    image_nmes = []        # per-image scalar NME (mean over visible landmarks)

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            image = sample["image"].unsqueeze(0).to(device)
            gt_heatmaps = sample["heatmaps"]
            visible = sample["visible"].numpy()

            pred_logits = model(image)
            pred_heatmaps = torch.sigmoid(pred_logits)[0].cpu()

            gt_coords = heatmaps_to_coords(gt_heatmaps)
            pred_coords = heatmaps_to_coords(pred_heatmaps)

            per_landmark_norm_error = compute_pck(pred_coords, gt_coords, visible,
                                                   landmark_order, norm_pair, PCK_THRESHOLDS)
            if per_landmark_norm_error is None:
                continue  # normalization landmarks not visible this image — excluded

            for c, name in enumerate(landmark_order):
                if visible[c] == 1:
                    all_errors.append((i, name, per_landmark_norm_error[c]))

            img_nme = compute_nme(pred_coords, gt_coords, visible, landmark_order, norm_pair)
            if img_nme is not None:
                image_nmes.append(img_nme)

    err_df = pd.DataFrame(all_errors, columns=["image_idx", "landmark", "normalized_error"])

    # --- Per-landmark stats ---
    landmark_stats = err_df.groupby("landmark")["normalized_error"].agg(
        mean="mean", median="median", std="std",
        q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
        n="count"
    ).reset_index()
    landmark_stats["iqr"] = landmark_stats["q75"] - landmark_stats["q25"]

    for t in PCK_THRESHOLDS:
        pck_per_landmark = err_df.groupby("landmark")["normalized_error"].apply(
            lambda x: (x <= t).mean()
        )
        landmark_stats[f"pck@{t}"] = landmark_stats["landmark"].map(pck_per_landmark)

    landmark_stats.insert(0, "view_family", view_family)

    # --- Per-view (aggregate) stats ---
    view_row = {
        "view_family": view_family,
        "n_test_images": len(test_ds),
        "n_evaluated_images": len(image_nmes),
        "mean_NME": np.mean(image_nmes),
        "median_NME": np.median(image_nmes),
        "std_NME": np.std(image_nmes),
        "q25_NME": np.quantile(image_nmes, 0.25),
        "q75_NME": np.quantile(image_nmes, 0.75),
    }
    for t in PCK_THRESHOLDS:
        view_row[f"pck@{t}"] = (err_df["normalized_error"] <= t).mean()

    return landmark_stats, view_row, err_df


def main():
    all_landmark_stats = []
    all_view_rows = []
    all_raw_errors = []

    for config_path in CONFIG_PATHS:
        landmark_stats, view_row, err_df = evaluate_view(config_path)
        all_landmark_stats.append(landmark_stats)
        all_view_rows.append(view_row)
        err_df["view_family"] = view_row["view_family"]
        all_raw_errors.append(err_df)

    landmark_table = pd.concat(all_landmark_stats, ignore_index=True)
    view_table = pd.DataFrame(all_view_rows)
    raw_errors_table = pd.concat(all_raw_errors, ignore_index=True)

    out_dir = Path("/content/drive/MyDrive/rhino-landmarks-data/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    landmark_table.to_csv(out_dir / "test_results_per_landmark.csv", index=False)
    view_table.to_csv(out_dir / "test_results_per_view.csv", index=False)
    raw_errors_table.to_csv(out_dir / "test_results_raw_errors.csv", index=False)

    print("\n" + "=" * 70)
    print("PER-VIEW SUMMARY")
    print("=" * 70)
    print(view_table.to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-LANDMARK SUMMARY")
    print("=" * 70)
    print(landmark_table.to_string(index=False))

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()