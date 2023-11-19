import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    @torch.inference_mode()
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate dice loss.

        Args:
            y_pred (torch.Tensor): NxCxHxW
            y_true (torch.Tensor): NxHxW

        Returns:
            torch.Tensor: dice loss (scalar)
        """
        assert y_true.dtype in (
            torch.int64,
            torch.int32,
        ), f"y_true.dtype: {y_true.dtype}"
        y_pred = y_pred.softmax(dim=1)
        num_classes = y_pred.shape[1]
        y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))  # NXC
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(
            y_true_one_hot, dim=(2, 3)
        )  # NXC
        dice_coefficient = (2 * intersection + self.smooth) / (
            union + self.smooth
        )  # NXC
        dice_loss = 1 - dice_coefficient.mean()
        return dice_loss
