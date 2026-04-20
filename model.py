import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional and pooling layers
        self.conv_pool = nn.Sequential(
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
        )

        # flatten the data
        self.flatten = nn.Flatten()

        # fully connected layers
        self.fully_conn = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_pool(x)
        x = self.flatten(x)
        logits = self.fully_conn(x)
        return logits

class LeNet_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Dropout(p=0.18),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Dropout(p=0.18)
        )
        self.flatten = nn.Flatten()
        self.fully_conn = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_pool(x)
        x = self.flatten(x)
        logits = self.fully_conn(x)
        return logits
