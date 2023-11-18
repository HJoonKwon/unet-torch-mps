import torch
from torchmetrics import JaccardIndex


def calculate_mean_iou(preds, targets):
    """
    Mean Intersection over Union (mIoU) calculation.

    Args:
    preds (torch.Tensor): Predicted segmentation map, shape [N, C, H, W].
    targets (torch.Tensor): Ground truth segmentation map, shape [N, H, W].

    Returns:
    float: Mean IoU score.
    """

    num_classes = preds.shape[1]
    preds = torch.argmax(preds, dim=1).view(-1)
    targets = targets.view(-1)

    # Create confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    conf_matrix = conf_matrix.index_add_(
        0, targets, torch.nn.functional.one_hot(preds, num_classes).float()
    )

    # IoU for each class
    intersection = torch.diag(conf_matrix)
    total = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = intersection / total.clamp(min=1e-6)

    # Mean IoU
    miou = torch.mean(iou[total > 0])  # Mean over classes with non-zero total
    return miou.item()


class IoU(torch.nn.Module):
    def __init__(self, num_classes: int, task="multiclass", per_class=False):
        super().__init__()
        assert (num_classes > 2 and task == "multiclass") or (
            num_classes == 2 and task == "binary"
        )
        if per_class:
            self.iou = JaccardIndex(num_classes=num_classes, task=task, average=None)
        else:
            self.iou = JaccardIndex(num_classes=num_classes, task=task)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.iou(pred, target)
