import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.projection = (stride != 1) or (in_channels != out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.batchNorm1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.batchNorm2 = nn.BatchNorm2d(num_features=self.out_channels)

        self.leakyReLU = nn.LeakyReLU()

        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.leakyReLU(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)

        x += identity

        x = self.leakyReLU(x)

        return x


# test block

block = Block(in_channels=64, out_channels=128, stride=2)

x = torch.randn(1, 64, 56, 56)

output = block(x)

print(output.shape)
