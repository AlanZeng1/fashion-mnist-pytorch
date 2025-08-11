import torch 
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(in_features=24*3*3, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.flatten(start_dim = 1)
        t = self.fc1(t)
        t = F.relu(t)
        
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        
        return t
