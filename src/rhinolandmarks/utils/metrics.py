import numpy as np
import torch


def heatmaps_to_coords(heatmaps):
    """
    heatmaps: (C, H, W) numpy array or tensor
    Returns: (C, 2) array of (x, y) pixel coords — argmax location per channel.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    C, H, W = heatmaps.shape
    coords = np.zeros((C, 2), dtype=np.float32)
    for c in range(C):
        idx = np.argmax(heatmaps[c])
        y, x = np.unravel_index(idx, (H, W))
        coords[c] = [x, y]
    return coords


def compute_nme(pred_coords, gt_coords, visible, landmark_order, norm_pair):
    """
    pred_coords, gt_coords: (C, 2) arrays, same coordinate space
    visible: (C,) 1.0/0.0 array
    landmark_order: list of landmark names matching channel order
    norm_pair: (name_a, name_b) landmarks defining normalization distance

    Returns scalar NME for this image, or None if it can't be computed
    (normalization landmarks missing, or degenerate zero distance).
    """
    idx_a = landmark_order.index(norm_pair[0])
    idx_b = landmark_order.index(norm_pair[1])

    if visible[idx_a] == 0 or visible[idx_b] == 0:
        return None

    norm_dist = np.linalg.norm(gt_coords[idx_a] - gt_coords[idx_b])
    if norm_dist < 1e-6:
        return None

    per_landmark_error = np.linalg.norm(pred_coords - gt_coords, axis=1)
    visible_mask = visible.astype(bool)

    if visible_mask.sum() == 0:
        return None

    return per_landmark_error[visible_mask].mean() / norm_dist