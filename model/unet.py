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
        self.layers.append(UpConv(1024, 1024))
        self.layers.append(Conv3(1024, 512))
        self.layers.append(Conv3(512, 512))

        self.layers.append(UpConv(512, 512))
        self.layers.append(Conv3(512, 256))
        self.layers.append(Conv3(256, 256))

        self.layers.append(UpConv(256, 256))
        self.layers.append(Conv3(256, 128))
        self.layers.append(Conv3(128, 128))

        self.layers.append(UpConv(128, 128))
        self.layers.append(Conv3(128, 64))
        self.layers.append(Conv3(64, 64))
        self.layers.append(Conv1(64, out_ch))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)
