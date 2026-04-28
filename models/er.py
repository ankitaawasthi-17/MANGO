
import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.buffer import ReplayBuffer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ER(nn.Module):
    def __init__(self, num_classes=10, buffer_size=2000, lr=0.05):
        super().__init__()

        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()

        self.buffer = ReplayBuffer(buffer_size)

    def forward(self, x):
        return self.net(x)

    def observe(self, x, y):
        self.train()

        bx, by = self.buffer.sample(x.size(0))

        if bx is not None:
            x = torch.cat([x, bx])
            y = torch.cat([y, by])

        logits = self(x)
        loss = self.loss(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.add(x, y)
