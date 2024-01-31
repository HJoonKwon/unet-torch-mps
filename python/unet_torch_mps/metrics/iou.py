import torch.nn.functional as F


def calculate_miou(pred, target):
    """
    Mean Intersection over Union (mIoU) calculation.

    Args:
    pred (torch.Tensor): Predicted segmentation map, shape [N, C, H, W].
    target (torch.Tensor): Ground truth segmentation map, shape [N, H, W].

    Returns:
    float: Mean IoU score.
    """

    pred_argmax = F.softmax(pred, dim=1).argmax(1)
    iou = (pred_argmax == target.squeeze()).float().mean()
    return iou.item()
