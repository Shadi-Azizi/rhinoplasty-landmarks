import argparse
import random
from pathlib import Path
import csv
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset

from rhinolandmarks.datasets.heatmap_dataset import ViewLandmarkDataset
from rhinolandmarks.datasets.splits import get_split_json_paths
from rhinolandmarks.models.registry import build_model
from rhinolandmarks.losses.heatmap_losses import MaskedHeatmapMSELoss
from rhinolandmarks.utils.metrics import heatmaps_to_coords, compute_nme
from rhinolandmarks.taxonomy import landmarks_for, NORM_LANDMARK_PAIRS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg, split):
    """Merges json paths across all view_folders (e.g. lateral_left + lateral_right)
    belonging to this split, into one ViewLandmarkDataset."""
    data_root = Path(cfg["data_root"])
    split_path = Path(cfg["split_path"])

    all_json_paths = []
    for folder_name in cfg["view_folders"]:
        folder = data_root / folder_name
        all_json_paths.extend(get_split_json_paths(folder, split, split_path))

    return ViewLandmarkDataset(
        json_paths=all_json_paths,
        image_size=tuple(cfg["image_size"]),
        heatmap_size=tuple(cfg["heatmap_size"]),
        sigma=cfg["heatmap_sigma"],
    )


def evaluate(model, loader, loss_fn, device, view_family, landmark_order, norm_pair):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    nme_values = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)
            visible = batch["visible"].to(device)

            preds = model(images)
            loss = loss_fn(preds, targets, visible)
            total_loss += loss.item()
            n_batches += 1

            preds_sigmoid = torch.sigmoid(preds)
            for i in range(images.shape[0]):
                gt_coords = heatmaps_to_coords(targets[i])
                pred_coords = heatmaps_to_coords(preds_sigmoid[i])
                vis_np = batch["visible"][i].numpy()
                nme = compute_nme(pred_coords, gt_coords, vis_np, landmark_order, norm_pair)
                if nme is not None:
                    nme_values.append(nme)

    avg_loss = total_loss / max(n_batches, 1)
    avg_nme = float(np.mean(nme_values)) if nme_values else float("inf")
    return avg_loss, avg_nme


def main(config_path, resume=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. On Colab, set Runtime -> Change runtime type -> GPU.")

    view_family = cfg["view_family"]
    landmark_order = landmarks_for(view_family)
    norm_pair = NORM_LANDMARK_PAIRS[view_family]
    num_channels = len(landmark_order)

    print(f"Training view_family='{view_family}', channels={num_channels}")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    model_name = cfg.get("model_name", "unet")  # defaults to unet if not specified, for backward compatibility
    model = build_model(model_name, in_channels=3, out_channels=num_channels,
                         base_channels=cfg["base_channels"]).to(device)

    loss_fn = MaskedHeatmapMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history_path = checkpoint_dir / f"{model_name}_{view_family}_history.csv"
    history_mode = "a" if (resume and history_path.exists()) else "w"
    history_file = open(history_path, history_mode, newline="")
    history_writer = csv.writer(history_file)
    if history_mode == "w":
        history_writer.writerow(["epoch", "train_loss", "val_loss", "val_nme", "lr"])
    best_ckpt_path = checkpoint_dir / f"{model_name}_{view_family}_best.pt"

    start_epoch = 1
    best_val_nme = float("inf")
    epochs_without_improvement = 0

    if resume and best_ckpt_path.exists():
        print(f"Resuming from checkpoint: {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_nme = checkpoint["val_nme"]
        print(f"  Resuming at epoch {start_epoch}, best_val_NME so far: {best_val_nme:.4f}")
    elif resume:
        print("--resume was set but no checkpoint found. Starting fresh.")


    

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        model.train()
        train_loss_total = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)
            visible = batch["visible"].to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, targets, visible)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_total / max(n_batches, 1)
        val_loss, val_nme = evaluate(model, val_loader, loss_fn, device,
                                      view_family, landmark_order, norm_pair)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg['num_epochs']}  "
              f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_NME={val_nme:.4f}  lr={current_lr:.2e}")
        history_writer.writerow([epoch, avg_train_loss, val_loss, val_nme, current_lr])
        history_file.flush()

        if val_nme < best_val_nme:
            best_val_nme = val_nme
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_nme": val_nme,
                "config": cfg,
            }, best_ckpt_path)
            print(f"  -> New best val_NME={val_nme:.4f}, checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg['early_stopping_patience']} epochs).")
                break

    history_file.close()
    print(f"Training complete. Best val_NME={best_val_nme:.4f}, checkpoint at {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if it exists")
    args = parser.parse_args()
    main(args.config, resume=args.resume)