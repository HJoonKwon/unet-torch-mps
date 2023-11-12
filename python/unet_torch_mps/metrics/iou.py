import torch
from torchmetrics import JaccardIndex


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
