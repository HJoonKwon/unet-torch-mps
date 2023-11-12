import torch 
from unet_torch_mps.metrics.iou import IoU 

def test_IoU():
    
    pred = torch.Tensor([[[[1, 2], [1, 1]]]])
    target = torch.Tensor([[[[1, 1], [1, 1]]]])
    
    iou = IoU(num_classes=3, task="multiclass", per_class=False)
    assert iou(pred, target) == 0.375 
    
    iou = IoU(num_classes=3, task="multiclass", per_class=True)
    assert torch.allclose(iou(pred, target), torch.Tensor([0.0, 0.75, 0.0]))