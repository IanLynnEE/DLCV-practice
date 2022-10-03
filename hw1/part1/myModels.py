import torch
import torch.nn as nn


class myLeNet(nn.Module):
    def __init__(self, num_out):
        print('Initializing myLeNet...')
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(18, 6, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(nn.Linear(400, 200), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200, 200), nn.ReLU())
        self.fc3 = nn.Linear(200, num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class myAlexNet(nn.Module):
    def __init__(self, num_out):
        print('Initializing myAlexNet...')
        super(myAlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc1 = nn.Sequential(nn.Linear(256*3*3, 1024), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc3 = nn.Linear(256, num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
