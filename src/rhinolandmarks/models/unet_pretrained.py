"""
U-Net with an ImageNet-pretrained ResNet34 encoder, otherwise identical
in spirit to unet.py: stride-4 output (matching heatmap target resolution),
skip connections at each resolution level, same decoder pattern.

Only the ENCODER changes — ResNet34's stem + 3 residual stages replace
the from-scratch DoubleConv/Down stack. Decoder channel widths are
adjusted to match ResNet34's actual stage output channels (64/64/128/256),
not simply reused from the from-scratch UNet.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class DoubleConv(nn.Module):
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


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        assert x.shape[-2:] == skip.shape[-2:], (
            f"Shape mismatch in skip connection: {x.shape} vs {skip.shape}."
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """
    Encoder: ImageNet-pretrained ResNet34 stem + layer1/layer2/layer3
    (stopping before layer4, since we only need down to 1/16 resolution
    to match the from-scratch U-Net's bottleneck depth).

    Resolution / channel map (ResNet34, input 512x640):
        stem (conv1+bn+relu):        stride 2  ->  64 ch   (1/2 res)
        maxpool + layer1:            stride 4  ->  64 ch   (1/4 res)  <- skip for final decoder stage
        layer2:                      stride 8  ->  128 ch  (1/8 res)  <- skip
        layer3:                      stride 16 -> 256 ch   (1/16 res) <- bottleneck

    Decoder mirrors unet.py's structure: upsample 1/16 -> 1/8 -> 1/4,
    output at stride 4 (matches heatmap target resolution), same as
    the from-scratch UNet — NOT upsampling further to full resolution.
    """

    def __init__(self, in_channels=3, out_channels=5, pretrained=True):
        super().__init__()
        assert in_channels == 3, "Pretrained ResNet34 expects 3-channel RGB input."

        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet34(weights=weights)

        # Stem: conv1 -> bn1 -> relu  (stride 2, 64 ch)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool          # stride 2 more -> total stride 4

        self.layer1 = resnet.layer1            # stride 4,  64 ch
        self.layer2 = resnet.layer2            # stride 8,  128 ch
        self.layer3 = resnet.layer3            # stride 16, 256 ch
        # layer4 (stride 32, 512 ch) intentionally omitted — not needed for stride-4 output

        # Decoder: mirrors unet.py's Up blocks, channel counts matched to ResNet34's actual widths
        self.up1 = Up(in_channels=256, skip_channels=128, out_channels=128)  # 1/16 -> 1/8
        self.up2 = Up(in_channels=128, skip_channels=64, out_channels=64)    # 1/8 -> 1/4

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)          # 1/2 res, 64 ch (not used as a skip — no 1/2-res decoder stage, matching original U-Net's stride-4 output convention)
        x1 = self.maxpool(x0)      # 1/4 res, 64 ch
        x1 = self.layer1(x1)       # 1/4 res, 64 ch   <- skip
        x2 = self.layer2(x1)       # 1/8 res, 128 ch  <- skip
        x3 = self.layer3(x2)       # 1/16 res, 256 ch <- bottleneck

        x = self.up1(x3, x2)       # -> 1/8 res
        x = self.up2(x, x1)        # -> 1/4 res

        return self.outc(x)        # (B, out_channels, H/4, W/4) — matches heatmap target size