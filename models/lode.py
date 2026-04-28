import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.buffer import ReplayBuffer
from models.er import ER
class LODE(ER):
    def observe(self, x, y):
        logits = self(x)

        loss_new = self.loss(logits, y)

        bx, by = self.buffer.sample(x.size(0))
        loss_old = 0

        if bx is not None:
            logits_b = self(bx)
            loss_old = self.loss(logits_b, by)

        loss = loss_new + 0.5 * loss_old

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.add(x, y)
