import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.buffer import ReplayBuffer
from models.er import ER
class GDumb(nn.Module):
    def __init__(self, num_classes=10, buffer_size=2000, lr=0.05):
        super().__init__()

        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.buffer = ReplayBuffer(buffer_size)
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def observe(self, x, y):
        self.buffer.add(x, y)

    def train_from_buffer(self):
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)

        for _ in range(1):  # 1 epoch
            bx, by = self.buffer.sample(256)
            logits = self(bx)
            loss = nn.CrossEntropyLoss()(logits, by)

            opt.zero_grad()
            loss.backward()
            opt.step()
