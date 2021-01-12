import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet50


class DummyBackbone(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        x = self.act(self.hidden(x))
        y_hat = self.act(self.fc(x))
        if y is None:
            return (y_hat,)
        loss = F.cross_entropy(y_hat, y)
        return (loss, y_hat)


class ResnetBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, freeze_resnet: bool = False):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(self.resnet.fc.in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_dim, out_features=num_classes),
        )
        self.model = nn.Sequential(
            nn.Sequential(*list(self.resnet.children())[:-2]),
            self.head
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        y_hat = self.model(x)
        if y is None:
            return (y_hat,)
        self.y_hat = y_hat
        loss = self.criterion(y_hat, y)
        return (loss, y_hat)
