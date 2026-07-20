import numpy as np


def generate_gaussian_heatmap(center, heatmap_size, sigma):
    """
    center: (x, y) in HEATMAP pixel space (already scaled down)
    heatmap_size: (w, h)
    Returns a (h, w) float32 array, peak value 1.0 at center.
    """
    w, h = heatmap_size
    cx, cy = center

    xs = np.arange(0, w, 1, dtype=np.float32)
    ys = np.arange(0, h, 1, dtype=np.float32)[:, None]

    heatmap = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def build_heatmap_stack(parsed, image_size, heatmap_size, sigma):
    """
    parsed: output of parse_labelme_json() — already flip-corrected,
            coordinates still in ORIGINAL image pixel space.
    image_size: (w, h) network input size
    heatmap_size: (w, h) heatmap resolution
    Returns:
        heatmaps: (C, heatmap_h, heatmap_w) float32 array
        visible:  (C,) float32 array (1.0 = include in loss, 0.0 = mask out)
    """
    orig_size = (parsed["image_width"], parsed["image_height"])
    landmark_order = parsed["landmark_order"]
    C = len(landmark_order)
    hw, hh = heatmap_size

    heatmaps = np.zeros((C, hh, hw), dtype=np.float32)
    visible = np.zeros((C,), dtype=np.float32)

    # Scale directly from ORIGINAL pixel space to HEATMAP pixel space —
    # no need to route through the intermediate image_size at all, since
    # it's the same linear scaling either way.
    scaled_points = {
        name: (x * hw / orig_size[0], y * hh / orig_size[1])
        for name, (x, y) in parsed["points"].items()
    }

    for i, name in enumerate(landmark_order):
        if parsed["visible"][name]:
            heatmaps[i] = generate_gaussian_heatmap(scaled_points[name], heatmap_size, sigma)
            visible[i] = 1.0
        # else: leave as zero heatmap, visible[i] stays 0.0 —
        # the training loop must multiply loss by `visible`, not just
        # rely on the zero heatmap being "harmless." A zero heatmap with
        # unmasked loss actively trains the network to predict nothing
        # everywhere, which is wrong.

    return heatmaps, visible