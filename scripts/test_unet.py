import torch
from rhinolandmarks.models.registry import build_model

VIEW_CHANNELS = {
    "frontal": 6,
    "basal": 2,
    "lateral": 5,
    "oblique": 5,
    "superior": 4,
}

IMAGE_SIZE = (512, 640)   # (w, h)
EXPECTED_HEATMAP_SIZE = (128, 160)  # (w, h) — must match heatmap_utils.py config

for view, out_channels in VIEW_CHANNELS.items():
    model = build_model("unet", in_channels=3, out_channels=out_channels, base_channels=32)
    model.eval()

    dummy_input = torch.randn(2, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (B, C, H, W)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (2, out_channels, EXPECTED_HEATMAP_SIZE[1], EXPECTED_HEATMAP_SIZE[0])
    status = "OK" if tuple(output.shape) == expected_shape else "MISMATCH"

    print(f"{view:10s} out_channels={out_channels}  output={tuple(output.shape)}  expected={expected_shape}  [{status}]")