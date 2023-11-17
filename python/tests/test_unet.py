import torch
from unet_torch_mps.model.unet import Unet

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def test_Unet():
    input = torch.randn((1, 1, 512, 512), device=device)
    model = Unet(1, 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 512, 512)

    input = torch.randn((1, 1, 378, 378), device=device)
    model = Unet(1, 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 378, 378)
