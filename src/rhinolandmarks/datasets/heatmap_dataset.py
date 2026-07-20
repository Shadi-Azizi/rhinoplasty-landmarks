from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from .labelme_parser import parse_labelme_json
from ..utils.heatmap_utils import build_heatmap_stack
from ..taxonomy import landmarks_for


class ViewLandmarkDataset(Dataset):
    """
    One dataset instance = one view family (e.g. all lateral_left +
    lateral_right images together, since they're mirror-pairs of the
    same optical feature). Each view family gets its OWN model, trained
    separately — this dataset just needs to guarantee it only ever
    contains images from a single view family.

    Returns per item:
        image:    (3, H, W) float32 tensor, normalized to [0, 1]
        heatmaps: (C, hh, hw) float32 tensor
        visible:  (C,) float32 tensor (1.0 = include in loss)
    """

    def __init__(self, json_paths, image_size, heatmap_size, sigma):
        """
        json_paths: list of Path/str to labelme JSONs, ALL belonging to
                    the same view family. Build this list per-view-family
                    in a separate splitting script — this class doesn't
                    do folder scanning itself, to keep it testable and
                    decoupled from your disk layout.
        image_size: (w, h) network input size, e.g. (512, 640)
        heatmap_size: (w, h) heatmap output size, e.g. (128, 160)
        sigma: gaussian sigma in heatmap pixel space
        """
        self.json_paths = [Path(p) for p in json_paths]
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        if len(self.json_paths) == 0:
            raise ValueError("Received an empty list of json_paths.")

        # Sanity-check: confirm every file belongs to the SAME view family.
        # Catching a mixed-family list here, at construction time, is far
        # cheaper than discovering it as a shape-mismatch crash deep
        # inside a DataLoader worker later.
        first_parsed = parse_labelme_json(self.json_paths[0])
        self.view_family = first_parsed["view_family"]
        self.num_channels = len(landmarks_for(self.view_family))

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        parsed = parse_labelme_json(json_path)

        if parsed["view_family"] != self.view_family:
            raise ValueError(
                f"{json_path} has view family '{parsed['view_family']}', "
                f"expected '{self.view_family}'. This dataset must contain "
                f"only one view family."
            )

        # Derive image path from the JSON's own filename, since json/image
        # names match exactly — more reliable than trusting the JSON's
        # internal imagePath field, which can be stale or relative.
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            for ext in [".jpeg", ".JPG", ".png"]:
                alt = json_path.with_suffix(ext)
                if alt.exists():
                    image_path = alt
                    break

        img = Image.open(image_path).convert("RGB")

        if parsed["flipped"]:
            # Match the coordinate mirroring already applied in the parser
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize(self.image_size, Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0    # (H, W, 3)
        img_arr = np.transpose(img_arr, (2, 0, 1))            # (3, H, W)

        heatmaps, visible = build_heatmap_stack(
            parsed, self.image_size, self.heatmap_size, self.sigma
        )

        return {
            "image": torch.from_numpy(img_arr),
            "heatmaps": torch.from_numpy(heatmaps),
            "visible": torch.from_numpy(visible),
            "view_family": self.view_family,
            "json_path": str(json_path),   # kept for debugging/visualization later
        }