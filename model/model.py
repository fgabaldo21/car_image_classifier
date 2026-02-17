import torch
import torch.nn as nn
import yaml

from model.block import Block

with open("./config/config.yaml", "r") as f:
    data = yaml.safe_load(f)


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.batchNorm = nn.BatchNorm2d(num_features=64)
        self.activation = nn.LeakyReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            num_blocks=data["num_blocks"][0], out_channels=data["channels"][0], stride=1
        )
        self.layer2 = self._make_layer(
            num_blocks=data["num_blocks"][1], out_channels=data["channels"][1], stride=2
        )
        self.layer3 = self._make_layer(
            num_blocks=data["num_blocks"][2], out_channels=data["channels"][2], stride=2
        )
        self.layer4 = self._make_layer(
            num_blocks=data["num_blocks"][3], out_channels=data["channels"][3], stride=2
        )

        self.globalPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_features=512, out_features=196)

    def _make_layer(self, num_blocks: int, out_channels: int, stride: int):
        layers = []

        layers.append(
            Block(
                in_channels=self.in_channels, out_channels=out_channels, stride=stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                Block(in_channels=self.in_channels, out_channels=out_channels, stride=1)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.activation(x)
        x = self.maxPool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.globalPool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x


# test block

dummy_input = torch.randn(1, 3, 256, 256)

model = Cnn()

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
