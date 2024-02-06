import torch
import torch.nn as nn
import torch.nn.functional as F


class TanimotoLoss(nn.Module):
    def __init__(self, smooth=1e-5, axis=(2, 3)):
        super().__init__()
        self.axis = axis
        self.smooth = smooth

    def forward(self, preds, label_one_hot):
        # Calculate the class volume per batch
        Vli = torch.mean(torch.sum(label_one_hot, dim=self.axis), dim=0)
        wli = torch.reciprocal(Vli**2 + self.smooth)  # Avoid division by zero

        # Handle infinite weights
        new_weights = torch.where(torch.isinf(wli), torch.zeros_like(wli), wli)
        wli = torch.where(
            torch.isinf(wli), torch.ones_like(wli) * torch.max(new_weights), wli
        )

        # Compute Tanimoto coefficient
        rl_x_pl = torch.sum(label_one_hot * preds, dim=self.axis)
        l = torch.sum(label_one_hot**2, dim=self.axis)
        p = torch.sum(preds**2, dim=self.axis)
        rl_p_pl = l + p - rl_x_pl

        tnmt = (torch.sum(wli * rl_x_pl, dim=1) + self.smooth) / (
            torch.sum(wli * rl_p_pl, dim=1) + self.smooth
        )

        return tnmt


class TanimotoWithDualLoss(nn.Module):
    """
    Tanimoto coefficient with dual form: Diakogiannis et al 2019
    """

    def __init__(self, num_classes, smooth=1e-5, axis=(2, 3), weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.tanimoto_loss = TanimotoLoss(smooth, axis)

    def forward(self, preds_softmax, label):
        # Convert label to one-hot encoding
        label_one_hot = (
            torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Measure of overlap using the softmax probabilities and one-hot labels
        loss1 = self.tanimoto_loss(preds_softmax, label_one_hot)

        # Measure of non-overlap
        preds_dual = 1.0 - preds_softmax
        labels_dual = 1.0 - label_one_hot
        loss2 = self.tanimoto_loss(preds_dual, labels_dual)

        return 1 - 0.5 * (loss1 + loss2).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

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
        if y_true.dtype == torch.int32:
            y_true = y_true.long()
        assert len(y_true.shape) == 3, f"y_true.shape: {y_true.shape}"

        y_pred = y_pred.softmax(dim=1)
        num_classes = y_pred.shape[1]
        assert (
            y_true.min() >= 0 and y_true.max() < num_classes
        ), f"Invalid values in y_true: min {y_true.min()}, max {y_true.max()}"
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
