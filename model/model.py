import torch.nn as nn
import torch

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.batchNorm = nn.BatchNorm2d(num_features=32, momentum=0.1)
        self.drop = nn.Dropout2d(p=0.25)

        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten(start_dim=1)

        self.linear = nn.Linear(in_features=15366400, out_features=196)

    def forward(self, x):
        x = self.conv_block1(x)

        x = self.batchNorm(x)
        x = self.drop(x)

        x = self.conv_block2(x)

        x = self.flatten(x)

        x = self.linear(x)
        x = self.sigmoid(x)

        return x

# test block

dummy_input = torch.randn(1, 1, 256, 256)

model = Cnn()

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)