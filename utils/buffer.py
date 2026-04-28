import random
import torch

class ReplayBuffer:
    def __init__(self, capacity, device=None):
        self.capacity = capacity
        self.data = []
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def add(self, x, y):
        x = x.detach().cpu()
        y = y.detach().cpu()

        for xi, yi in zip(x, y):
            if len(self.data) < self.capacity:
                self.data.append((xi, yi))
            else:
                idx = random.randint(0, self.capacity - 1)
                self.data[idx] = (xi, yi)

    def sample(self, batch_size):
        if len(self.data) == 0:
            return None, None

        batch = random.sample(self.data, min(batch_size, len(self.data)))
        x, y = zip(*batch)

        x = torch.stack(x).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        return x, y

    def __len__(self):
        return len(self.data)
