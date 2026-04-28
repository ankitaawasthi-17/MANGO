# train/trainer_clear.py

import numpy as np
import torch
import random
from torchvision import transforms

from data.task_split_clear import get_task_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyperparameters ───────────────────────────────────────────────────────────
ONLINE_BATCH    = 32
REPLAY_BATCH    = 64
META_BATCH_SIZE = 128
META_EVERY      = 3     # meta-update every N batches

# CLEAR images are already 64×64 after task_split resize
_replay_aug = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
])


# ── Replay buffer (reservoir sampling) ───────────────────────────────────────

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.xs          = []
        self.ys          = []
        self.n_seen      = 0

    def add(self, x_batch, y_batch):
        x_batch = x_batch.cpu()
        y_batch = y_batch.cpu()
        for x, y in zip(x_batch, y_batch):
            self.n_seen += 1
            if len(self.xs) < self.buffer_size:
                self.xs.append(x)
                self.ys.append(int(y))
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.buffer_size:
                    self.xs[j] = x
                    self.ys[j] = int(y)

    def sample(self, n):
        if not self.xs:
            return None, None
        n   = min(n, len(self.xs))
        idx = random.sample(range(len(self.xs)), n)
        xs  = torch.stack([self.xs[i] for i in idx]).to(DEVICE)
        ys  = torch.tensor([self.ys[i] for i in idx],
                           dtype=torch.long).to(DEVICE)
        return xs, ys

    def __len__(self):
        return len(self.xs)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_task(model, task_id, classes_per_task):
    """
    CLEAR-10: all 10 classes exist in every task → TIL == CIL.
    Argmax over all 10 logits directly.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in get_task_loader(task_id, classes_per_task,
                                     train=False, batch_size=200):
            x, y  = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate(model, task, acc_til, acc_cil, cpt):
    accs = []
    for t in range(task + 1):
        a = _eval_task(model, t, cpt)
        acc_til[task, t] = a
        acc_cil[task, t] = a   # TIL == CIL on CLEAR
        accs.append(a)
    print(f"  ACC per task: {[round(a, 1) for a in accs]}")
    print(f"  Mean so far : {np.mean(accs):.2f}%")


def compute_final_metrics(acc_matrix, label=""):
    n   = acc_matrix.shape[0]
    acc = np.mean(acc_matrix[n-1, :n])
    bwt = np.mean([acc_matrix[n-1, t] - acc_matrix[t, t]
                   for t in range(n-1)])
    aaa = np.mean([np.mean(acc_matrix[t, :t+1]) for t in range(n)])
    wc  = np.mean([np.min(acc_matrix[t,  :t+1]) for t in range(n)])
    tag = f" [{label}]" if label else ""
    print(f"\n===== FINAL METRICS{tag} =====")
    print(f"  ACC   : {acc:.2f}")
    print(f"  AAA   : {aaa:.2f}")
    print(f"  WC-Acc: {wc:.2f}")
    print(f"  BWT   : {bwt:.2f}")
    return {"acc": acc, "aaa": aaa, "wc_acc": wc, "bwt": bwt}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_sequence(model, num_tasks, classes_per_task,
                   glances, epochs, buffer_size):

    method = getattr(model, "method", "ours")

    print(f"Method={method} | Buffer={buffer_size} | "
          f"OnlineBatch={ONLINE_BATCH} | ReplayBatch={REPLAY_BATCH} | "
          f"MetaEvery={META_EVERY}\n")

    acc_til = np.zeros((num_tasks, num_tasks))
    acc_cil = np.zeros((num_tasks, num_tasks))
    buf     = ReplayBuffer(buffer_size)

    for task in range(num_tasks):
        print(f"\n{'='*55}")
        print(f"  TASK {task}  (time bucket {task+1}/10)")
        print(f"{'='*55}")

        # Snapshot params for regularization (skip task 0)
        if task > 0 and hasattr(model, "save_old_params"):
            model.save_old_params()

        loader    = get_task_loader(task, classes_per_task,
                                    train=True, batch_size=ONLINE_BATCH)
        n_batches = 0

        for epoch in range(epochs):
            for _ in range(glances):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)

                    # ── Online update ─────────────────────────────────────────
                    model.observe(x, y)

                    # ── Replay ────────────────────────────────────────────────
                    if task > 0 and len(buf) >= REPLAY_BATCH:
                        x_rep, y_rep = buf.sample(REPLAY_BATCH)
                        model.observe(_replay_aug(x_rep), y_rep)

                        # ── Meta-update lambda (ours only) ────────────────────
                        if (method == "ours"
                                and n_batches % META_EVERY == 0
                                and hasattr(model, "meta_update_lambda")):
                            x_mem, y_mem = buf.sample(META_BATCH_SIZE)
                            if x_mem is not None:
                                model.meta_update_lambda(x, y, x_mem, y_mem)

                    buf.add(x, y)
                    n_batches += 1

        print(f"  Batches: {n_batches} | Buffer: {len(buf)}")

        if method == "ours" and hasattr(model, "lambdas"):
            lams  = model.lambdas.detach().cpu().tolist()
            names = ["stem+L1", "layer2", "layer3", "layer4", "fc"]
            print("  [λ: "
                  + "  ".join(f"{n}={v:.5f}" for n, v in zip(names, lams))
                  + "]")
        elif method == "fixed_lambda":
            from models.ocl_resnet18_tiny import FIXED_LAMBDA
            print(f"  [λ: fixed={FIXED_LAMBDA} for all layers]")

        print("\n--- Evaluation ---")
        evaluate(model, task, acc_til, acc_cil, classes_per_task)

    return acc_til, acc_cil
