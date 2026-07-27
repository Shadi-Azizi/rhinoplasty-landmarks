import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2 — the basic repeated block in U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale then DoubleConv — one encoder stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscale, concatenate skip connection, then DoubleConv — one decoder stage."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        assert x.shape[-2:] == skip.shape[-2:], (
            f"Shape mismatch in skip connection: {x.shape} vs {skip.shape}. "
            f"Check that image_size is divisible by 32."
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for heatmap regression, with a STRIDE-4 output (i.e. output
    spatial resolution = input resolution / 4), matching heatmap_utils.py's
    target resolution — this is standard practice in pose/landmark heatmap
    networks (c.f. Stacked Hourglass, HRNet), both to save compute and
    because sub-stride-4 pixel precision isn't meaningful given natural
    annotation noise.

    Encoder still goes down 4 levels (bottleneck at 1/16 resolution) for
    a large enough receptive field / semantic depth. Decoder only
    upsamples back 2 levels (1/16 -> 1/8 -> 1/4), stopping at stride 4
    instead of fully returning to input resolution.

    in_channels: 3 for RGB input
    out_channels: number of landmark heatmap channels — SET PER VIEW FAMILY
                  (6 frontal, 2 basal, 5 lateral, 5 oblique, 4 superior).
                  One model instance trained per view family.
    base_channels: width of first encoder stage; doubles each downsampling
                   stage (32 -> 64 -> 128 -> 256 -> 512 at bottleneck).
    """

    def __init__(self, in_channels=3, out_channels=5, base_channels=32):
        super().__init__()
        c = base_channels

        self.inc = DoubleConv(in_channels, c)          # level 0, full res
        self.down1 = Down(c, c * 2)                     # level 1, 1/2 res
        self.down2 = Down(c * 2, c * 4)                  # level 2, 1/4 res
        self.down3 = Down(c * 4, c * 8)                  # level 3, 1/8 res
        self.down4 = Down(c * 8, c * 16)                 # level 4, 1/16 res (bottleneck)

        self.up1 = Up(c * 16, c * 8, c * 8)               # 1/16 -> 1/8, skip=level3
        self.up2 = Up(c * 8, c * 4, c * 4)                # 1/8 -> 1/4,  skip=level2

        # Output at 1/4 resolution — matches heatmap_size (128x160 for 512x640 input)
        self.outc = nn.Conv2d(c * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)      # full res
        x2 = self.down1(x1)   # 1/2 res
        x3 = self.down2(x2)   # 1/4 res
        x4 = self.down3(x3)   # 1/8 res
        x5 = self.down4(x4)   # 1/16 res (bottleneck)

        x = self.up1(x5, x4)  # 1/8 res
        x = self.up2(x, x3)   # 1/4 res

        return self.outc(x)   # (B, out_channels, H/4, W/4) — raw logits, no activation