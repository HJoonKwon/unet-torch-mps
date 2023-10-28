import torch
from model.blocks import Conv1, Conv3, UpConv, MaxPool

device = "mps" if torch.backends.mps.is_available() else "cpu"


def test_Conv1():
    input = torch.randn((1, 64, 388, 388), device=device)
    model = Conv1(input.shape[1], 2).to(device)
    output = model(input)
    assert output.shape == (1, 2, 388, 388)


def test_Conv3():
    input = torch.randn((1, 1024, 32, 32), device=device)
    model = Conv3(input.shape[1], 1024).to(device)
    output = model(input)
    assert output.shape == (1, 1024, 30, 30)


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