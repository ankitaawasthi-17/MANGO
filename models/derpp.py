import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.buffer import ReplayBuffer
from models.er import ER
class DERPP(ER):
    def __init__(self, num_classes=10, buffer_size=2000, lr=0.05, alpha=0.5, beta=0.5):
        super().__init__(num_classes, buffer_size, lr)
        self.alpha = alpha
        self.beta = beta
        self.logits_buffer = []

    def observe(self, x, y):
        self.train()

        logits = self(x)
        loss = self.loss(logits, y)

        bx, by = self.buffer.sample(x.size(0))

        if bx is not None:
            logits_b = self(bx)
            loss += self.alpha * self.loss(logits_b, by)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.add(x, y)
