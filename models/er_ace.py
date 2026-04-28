from models.er import ER
class ER_ACE(ER):
    def __init__(self, num_classes=10, buffer_size=2000, lr=0.05):
        super().__init__(num_classes, buffer_size, lr)

    def observe(self, x, y):
        self.train()

        bx, by = self.buffer.sample(x.size(0))

        logits = self(x)

        # 🔥 mask logits to current labels
        unique_labels = y.unique()
        mask = torch.zeros_like(logits)
        mask[:, unique_labels] = 1

        loss = (self.loss(logits * mask, y))

        if bx is not None:
            logits_b = self(bx)
            loss += self.loss(logits_b, by)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.add(x, y)
