import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_torch_mps.model.blocks import Conv1, Conv3, UpConv, MaxPool


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

        self.num_layers_list = [2, 3, 3, 3, 4, 3, 3, 3, 3]
        assert sum(self.num_layers_list) == len(self.layers)

        self.blocks = []
        layer_i = 0
        for num_layers in self.num_layers_list:
            self.blocks.append(
                nn.Sequential(*self.layers[layer_i : layer_i + num_layers])
            )
            layer_i += num_layers
        assert layer_i == len(self.layers)
        self.model = nn.Sequential(*self.blocks)

    def pad_and_concat(self, x: torch.Tensor, cached):
        hx, wx = x.shape[2:]
        y = cached.pop()
        hy, wy = y.shape[2:]
        if hy > hx or wy > wx:
            dy = (hy - hx) // 2
            dx = (wy - wx) // 2
            x = F.pad(x, (dx, (wy - wx) - dx, dy, (hy - hx) - dy))
        assert y.shape == x.shape, f"{y.shape}, {x.shape}"
        x = torch.concat([y, x], dim=1)
        return x

    def forward(self, x: torch.Tensor):
        cached = []
        for i in range(4):
            x = self.blocks[i](x)
            cached.append(x)

        for i in range(4, 8):
            x = self.blocks[i](x)
            x = self.pad_and_concat(x, cached)

        assert not cached
        x = self.blocks[-1](x)

        return x
