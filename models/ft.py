import torch
import torch.nn as nn
from torchvision.models import resnet18

class FT(nn.Module):
    def __init__(self, num_classes=10, lr=0.05):
        super().__init__()

        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def observe(self, x, y):
        logits = self(x)
        loss = self.loss(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
