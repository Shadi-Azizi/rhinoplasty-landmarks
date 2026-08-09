"""
HRNet-W18,
ImageNet-pretrained via timm, used as the field-standard architecture
reference point for this study.

Only the highest-resolution (stride-4) feature map is used for the
landmark head — matching the pose-estimation convention of relying on
HRNet's high-resolution stream for dense prediction, and matching the
stride-4 output convention used by every other model in this study.
The exact channel count at stride 4 is read dynamically from timm's
feature_info rather than hardcoded, since it differs from the printed
W32 example and shouldn't be assumed.
"""
import torch
import torch.nn as nn
import timm


class HRNetW18(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, pretrained=True):
        super().__init__()
        assert in_channels == 3, "Pretrained HRNet-W18 expects 3-channel RGB input."

        self.backbone = timm.create_model(
            "hrnet_w18.ms_aug_in1k",
            pretrained=pretrained,
            features_only=True,
        )

        # Find the feature level with stride (reduction) == 4, matching
        # every other model's output-resolution convention in this study.
        reductions = self.backbone.feature_info.reduction()
        if 4 not in reductions:
            raise RuntimeError(
                f"No stride-4 feature level found in HRNet-W18 feature_info. "
                f"Available reductions: {reductions}"
            )
        self.stride4_idx = reductions.index(4)
        stride4_channels = self.backbone.feature_info.channels()[self.stride4_idx]

        print(f"[HRNetW18] Using feature level {self.stride4_idx} "
              f"(stride {reductions[self.stride4_idx]}, {stride4_channels} channels) "
              f"as the landmark head input.")

        self.head = nn.Conv2d(stride4_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        stride4_feat = features[self.stride4_idx]
        return self.head(stride4_feat)   # (B, out_channels, H/4, W/4)