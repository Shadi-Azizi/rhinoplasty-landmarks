"""
A scaled-down HRNet-style architecture: parallel multi-resolution branches
maintained throughout (never fully collapsed to a single bottleneck, unlike
U-Net), with repeated cross-resolution fusion. Reduced channel width and
stage count relative to the original HRNet-W32/W48 [cite: Sun et al., 2019],
appropriate for this dataset's scale (~150-300 images per view family).
Output at stride 4, matching heatmap_utils.py's target resolution — same
convention as the U-Net side, for a fair comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    """Residual block used within each resolution branch."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class FusionModule(nn.Module):
    """
    Exchanges information across all current resolution branches: each
    output branch is the sum of every input branch, resampled (via strided
    conv for downsampling, or upsample+1x1 conv for upsampling) to that
    output branch's resolution. This repeated fusion is HRNet's defining
    mechanism — every resolution stays informed by every other resolution
    at every stage, rather than only communicating through skip connections
    at fixed points like U-Net.
    """
    def __init__(self, channels_list):
        super().__init__()
        self.n = len(channels_list)
        self.channels_list = channels_list
        self.resample = nn.ModuleList()
        for i in range(self.n):
            row = nn.ModuleList()
            for j in range(self.n):
                if i == j:
                    row.append(nn.Identity())
                elif i > j:
                    # branch j (higher res) -> branch i (lower res): downsample via strided convs
                    ops = []
                    for _ in range(i - j):
                        ops.append(ConvBNReLU(channels_list[j] if _ == 0 else channels_list[i],
                                               channels_list[i], stride=2))
                    row.append(nn.Sequential(*ops))
                else:
                    # branch j (lower res) -> branch i (higher res): upsample + 1x1 conv
                    row.append(nn.Sequential(
                        nn.Conv2d(channels_list[j], channels_list[i], 1, bias=False),
                        nn.BatchNorm2d(channels_list[i]),
                    ))
            self.resample.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):
        outputs = []
        target_sizes = [b.shape[-2:] for b in branches]
        for i in range(self.n):
            fused = 0
            for j in range(self.n):
                if i == j:
                    fused = fused + branches[j]
                elif i > j:
                    fused = fused + self.resample[i][j](branches[j])
                else:
                    up = F.interpolate(branches[j], size=target_sizes[i], mode="bilinear", align_corners=False)
                    fused = fused + self.resample[i][j](up)
            outputs.append(self.relu(fused))
        return outputs


class HRNetLite(nn.Module):
    """
    3-branch (stride 4 / 8 / 16), 2-stage scaled-down HRNet.
    Output: heatmaps at stride 4, matching the U-Net side's output resolution.
    """
    def __init__(self, in_channels=3, out_channels=5, base_channels=32):
        super().__init__()
        c = base_channels
        self.channels = [c, c * 2, c * 4]  # stride 4, 8, 16 branch widths

        # Stem: downsample input to stride 4 (matches U-Net's target resolution convention)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, c, stride=2),
            ConvBNReLU(c, c, stride=2),
        )

        # Stage 1: single branch at stride 4
        self.stage1_blocks = nn.Sequential(BasicBlock(c), BasicBlock(c))

        # Transition to 2 branches (stride 4, stride 8)
        self.transition1 = nn.ModuleList([
            nn.Identity(),
            ConvBNReLU(c, c * 2, stride=2),
        ])
        self.stage2_blocks = nn.ModuleList([
            nn.Sequential(BasicBlock(c), BasicBlock(c)),
            nn.Sequential(BasicBlock(c * 2), BasicBlock(c * 2)),
        ])
        self.fusion2 = FusionModule([c, c * 2])

        # Transition to 3 branches (stride 4, 8, 16)
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            ConvBNReLU(c * 2, c * 4, stride=2),
        ])
        self.stage3_blocks = nn.ModuleList([
            nn.Sequential(BasicBlock(c), BasicBlock(c)),
            nn.Sequential(BasicBlock(c * 2), BasicBlock(c * 2)),
            nn.Sequential(BasicBlock(c * 4), BasicBlock(c * 4)),
        ])
        self.fusion3 = FusionModule([c, c * 2, c * 4])
        self.fusion3b = FusionModule([c, c * 2, c * 4])  # second fusion round for more mixing

        # Head: use only the highest-resolution (stride 4) branch for final prediction —
        # standard HRNet practice for heatmap output.
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)                       # stride 4
        x = self.stage1_blocks(x)

        branches = [self.transition1[0](x), self.transition1[1](x)]
        branches = [blk(b) for blk, b in zip(self.stage2_blocks, branches)]
        branches = self.fusion2(branches)

        branches = [self.transition2[0](branches[0]),
                    self.transition2[1](branches[1]),
                    self.transition2[2](branches[1])]
        branches = [blk(b) for blk, b in zip(self.stage3_blocks, branches)]
        branches = self.fusion3(branches)
        branches = self.fusion3b(branches)

        return self.head(branches[0])           # stride-4 branch, matches heatmap target size