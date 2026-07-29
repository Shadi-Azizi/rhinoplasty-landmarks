import torch
import torch.nn as nn


class MaskedHeatmapMSELoss(nn.Module):
    """MSE between sigmoid(pred) and target heatmaps, masked by visibility."""

    def forward(self, pred_logits, target_heatmaps, visible):
        pred = torch.sigmoid(pred_logits)
        per_channel_error = ((pred - target_heatmaps) ** 2).mean(dim=(2, 3))
        masked_error = per_channel_error * visible
        num_visible = visible.sum()
        if num_visible == 0:
            return masked_error.sum() * 0.0
        return masked_error.sum() / num_visible