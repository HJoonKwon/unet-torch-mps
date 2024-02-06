import torch
from unet_torch_mps.model.unet import Conv1, Conv3, UpConv, MaxPool
from unet_torch_mps.model.unet import Unet

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def test_Conv1():
    input = torch.randn((1, 64, 388, 388), device=device)
    model = Conv1(input.shape[1], 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 388, 388)


def test_Conv3():
    input = torch.randn((1, 1024, 32, 32), device=device)
    model = Conv3(input.shape[1], 1024).to(device)
    output = model(input)
    assert output.shape == (1, 1024, 32, 32)


def test_UpConv():
    input = torch.randn((1, 1024, 28, 28), device=device)
    model = UpConv(input.shape[1], 1024).to(device)
    output = model(input)
    assert output.shape == (1, 1024, 56, 56)


def test_MaxPool():
    input = torch.randn((1, 512, 64, 64), device=device)
    model = MaxPool()
    output = model(input)
    assert output.shape == (1, 512, 32, 32)


def test_Unet():
    input = torch.randn((1, 1, 512, 512), device=device)
    model = Unet(1, 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 512, 512)

    input = torch.randn((1, 1, 378, 378), device=device)
    model = Unet(1, 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 378, 378)
