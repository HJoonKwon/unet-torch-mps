import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class ResUnetBranch(nn.Module):
    def __init__(self, in_ch: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, padding='same', bias=False, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, 1, padding='same', bias=False , dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(in_ch)
        self.act2 = nn.ReLU()
        self.layer = nn.Sequential(
            self.batchnorm1,
            self.act1,
            self.conv1,
            self.batchnorm2,
            self.act2,
            self.conv2,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x 

class ResUnetBlock(nn.Module):
    def __init__(self, in_ch, dilations):
        super().__init__()
        self.branches = [ResUnetBranch(in_ch, dilation) for dilation in dilations]
    def forward(self, x):
        for branch in self.branches:
            x += branch(x)
        return x 


class Conv2DN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        return self.layer(x)

class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, same=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, padding='same', bias=False) if same is True else nn.Conv2d(in_ch, out_ch, 1, 2, bias=False)
    def forward(self, x):
        return self.conv(x)

class Combine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.act = nn.ReLU()
        self.conv2dn = Conv2DN(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.act(x1)
        x = torch.concat([x1, x2], dim=1)
        x = self.conv2dn(x)
        return x 

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2dn = Conv2DN(in_ch, out_ch)
    def forward(self, x):
        B, C, H, W = x.shape 
        x = F.interpolate(x, scale_factor=2.0)
        x = self.conv2dn(x)
        return x 
    
class ResUnet(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.first = Conv2DN(in_ch, 32)
        self.down1 = ResUnetBlock(32, [1, 3, 15, 31])
        self.down2 = nn.Sequential( 
            Conv2D(32, 64, False),
            ResUnetBlock(64, [1, 3, 15, 31]),
        )
        self.down3 = nn.Sequential(
            Conv2D(64, 128, False),
            ResUnetBlock(128, [1, 3, 15]),
        )
        self.down4 = nn.Sequential(
            Conv2D(128, 256, False),
            ResUnetBlock(256, [1, 3, 15]),
        )
        self.down5 = nn.Sequential(
            Conv2D(256, 512, False),
            ResUnetBlock(512, [1]),
        )
        self.middle = nn.Sequential(
            Conv2D(512, 1024, False),
            ResUnetBlock(1024, [1]),
            UpSample(1024, 512),
        )
        self.combine1 = Combine(512*2, 512)
        self.up1 = nn.Sequential(
            ResUnetBlock(512, [1]),
            UpSample(512, 256),
        )
        self.combine2 = Combine(256*2, 256)
        self.up2 = nn.Sequential(
            ResUnetBlock(256, [1]),
            UpSample(256, 128),
        )
        self.combine3 = Combine(128*2, 128)
        self.up3 = nn.Sequential(
            ResUnetBlock(128, [1]),
            UpSample(128, 64),
        )
        self.combine4 = Combine(64*2, 64)
        self.up4 = nn.Sequential(
            ResUnetBlock(64, [1]),
            UpSample(64, 32),
        )
        self.combine5 = Combine(32*2, 32)
        self.up5 = ResUnetBlock(32, [1])
        self.combine6 = Combine(32*2, 32)
        self.last = nn.Conv2d(32, num_classes, 1, 1)
        
    def forward(self, x):
        x = self.first(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u = self.middle(d5)
        u = self.combine1(u, d5)
        u = self.up1(u)
        u = self.combine2(u, d4)
        u = self.up2(u)
        u = self.combine3(u, d3)
        u = self.up3(u)
        u = self.combine4(u, d2)
        u = self.up4(u)
        u = self.combine5(u, d1)
        u = self.up5(u)
        u = self.combine6(u, x)

        y = self.last(u)
        return y 
