import torch
from model.unet import Unet

device = "mps" if torch.backends.mps.is_available() else "cpu"


def test_Unet():
    input = torch.randn((1, 1, 572, 572), device=device)
    model = Unet(1, 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 388, 388)
