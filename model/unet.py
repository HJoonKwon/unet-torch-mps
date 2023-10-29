import torch
import torch.nn as nn
from model.blocks import Conv1, Conv3, UpConv, MaxPool


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.layers = []

        # down
        self.layers.append(Conv3(in_ch, 64))
        self.layers.append(Conv3(64, 64))
        self.layers.append(MaxPool())

        self.layers.append(Conv3(64, 128))
        self.layers.append(Conv3(128, 128))
        self.layers.append(MaxPool())

        self.layers.append(Conv3(128, 256))
        self.layers.append(Conv3(256, 256))
        self.layers.append(MaxPool())

        self.layers.append(Conv3(256, 512))
        self.layers.append(Conv3(512, 512))
        self.layers.append(MaxPool())

        self.layers.append(Conv3(512, 1024))
        self.layers.append(Conv3(1024, 1024))

        # up
        self.layers.append(UpConv(1024, 512))
        self.layers.append(Conv3(1024, 512))
        self.layers.append(Conv3(512, 512))

        self.layers.append(UpConv(512, 256))
        self.layers.append(Conv3(512, 256))
        self.layers.append(Conv3(256, 256))

        self.layers.append(UpConv(256, 128))
        self.layers.append(Conv3(256, 128))
        self.layers.append(Conv3(128, 128))

        self.layers.append(UpConv(128, 64))
        self.layers.append(Conv3(128, 64))
        self.layers.append(Conv3(64, 64))
        self.layers.append(Conv1(64, out_ch))
        
        self.model = nn.Sequential(*self.layers)
    
    def crop_and_concat(self, x: torch.Tensor, cached):
        hx, wx = x.shape[2:]
        y = cached.pop()
        hy, wy = y.shape[2:]
        y_cropped = y[:, :, (hy-hx)//2:(hy-hx)//2+hx, (wy-wx)//2:(wy-wx)//2+wx]
        assert y_cropped.shape == x.shape 
        x = torch.concat([y_cropped, x], dim=1)
        return x 

    def forward(self, x: torch.Tensor):
        cached = [] 
        for i in range(2):
            layer = self.layers.pop(0)
            x = layer(x)
        cached.append(x)
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        cached.append(x)
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        cached.append(x)
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        cached.append(x)
        
        for i in range(4):
            layer = self.layers.pop(0)
            x = layer(x)
        x = self.crop_and_concat(x, cached)
        
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        x = self.crop_and_concat(x, cached)
        
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        x = self.crop_and_concat(x, cached)
        
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
        x = self.crop_and_concat(x, cached)
        
        for i in range(3):
            layer = self.layers.pop(0)
            x = layer(x)
         
        assert not self.layers 
        return x
