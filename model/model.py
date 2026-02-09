import torch.nn as nn
import torch

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=1)
        
        self.linear1 = nn.Linear(in_features=43264, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=196)
        
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten(start_dim=1)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.flatten(x)
        
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear3(x)
        
        return x

# test block

"""dummy_input = torch.randn(1, 3, 128, 128)

model = Cnn()

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)"""