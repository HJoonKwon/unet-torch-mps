import torch 
import torch.nn as nn

class Conv3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__() 
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 0)
        self.act = nn.ReLU()
        self.layer = nn.Sequential(
            self.conv,
            self.act
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
class Conv1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
    def forward(self, x: torch.Tensor):
        return self.upconv(x)

class MaxPool(nn.Module):
    def __init__(self):
        super().__init__() 
        self.maxpool = nn.MaxPool2d(2, 2)
    def forward(self, x: torch.Tensor):
        return self.maxpool(x)
        