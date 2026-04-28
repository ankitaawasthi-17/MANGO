# train/trainer_tinyimagenet.py

import numpy as np
import torch
import random
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META_BATCH_SIZE = 256
ONLINE_BATCH    = 32
REPLAY_BATCH    = 64


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


# ── Replay augmentation (dataset-aware) ──────────────────────────────────────

def _get_aug(dataset):
    if dataset == "cifar":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    return transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_task(model, task_id, classes_per_task, mode="til"):
    dataset = getattr(model, "dataset", "tiny")
    if dataset == "cifar":
        from data.task_split import get_task_loader
    else:
        from data.task_split_tinyimagenet import get_task_loader

    model.eval()
    correct, total = 0, 0
    start = task_id * classes_per_task
    end   = start + classes_per_task
    with torch.no_grad():
        for x, y in get_task_loader(task_id, classes_per_task,
                                     train=False, batch_size=200):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if mode == "til":
                preds = logits[:, start:end].argmax(1) + start
            else:
                preds = logits[:, :end].argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_all(model, task, acc_til, acc_cil, cpt):
    til_list, cil_list = [], []
    for t in range(task + 1):
        til = _eval_task(model, t, cpt, "til")
        cil = _eval_task(model, t, cpt, "cil")
        acc_til[task, t] = til
        acc_cil[task, t] = cil
        til_list.append(til)
        cil_list.append(cil)
    print(f"  TIL: {[round(a, 1) for a in til_list]}")
    print(f"  CIL: {[round(a, 1) for a in cil_list]}")


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

def train_sequence(model, num_tasks=20, classes_per_task=10,
                   glances=1, epochs=1, buffer_size=None):

    dataset = getattr(model, "dataset", "tiny")
    method  = getattr(model, "method",  "ours")

    if dataset == "cifar":
        BUFFER_SIZE = buffer_size if buffer_size is not None else 2000
        META_EVERY  = 3
    else:
        BUFFER_SIZE = buffer_size if buffer_size is not None else 4000
        META_EVERY  = 5

    if dataset == "cifar":
        import data.task_split as ts
    else:
        import data.task_split_tinyimagenet as ts
    get_task_loader = ts.get_task_loader

    replay_aug = _get_aug(dataset)

    print(f"Method={method} | Dataset={dataset} | Buffer={BUFFER_SIZE} | "
          f"OnlineBatch={ONLINE_BATCH} | ReplayBatch={REPLAY_BATCH} | "
          f"MetaEvery={META_EVERY}\n")

    acc_til = np.zeros((num_tasks, num_tasks))
    acc_cil = np.zeros((num_tasks, num_tasks))
    buf     = ReplayBuffer(BUFFER_SIZE)

    for task in range(num_tasks):
        print(f"\n{'='*55}")
        print(f"  TASK {task}  "
              f"(classes {task*classes_per_task}–"
              f"{(task+1)*classes_per_task - 1})")
        print(f"{'='*55}")

        if task > 0 and hasattr(model, "save_old_params"):
            model.save_old_params()

        loader    = get_task_loader(task, classes_per_task,
                                    train=True, batch_size=ONLINE_BATCH)
        n_batches = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # ── Online glances ────────────────────────────────────────────────
            for _ in range(glances):
                model.observe(x, y)

            # ── Replay (all methods use replay) ───────────────────────────────
            if task > 0 and len(buf) >= REPLAY_BATCH:
                x_rep, y_rep = buf.sample(REPLAY_BATCH)
                model.observe(replay_aug(x_rep), y_rep)

                # ── Meta-update lambda (ours only) ────────────────────────────
                if (method == "ours"
                        and n_batches % META_EVERY == 0
                        and hasattr(model, "meta_update_lambda")):
                    x_mem, y_mem = buf.sample(META_BATCH_SIZE)
                    if x_mem is not None:
                        model.meta_update_lambda(x, y, x_mem, y_mem)

            buf.add(x, y)
            n_batches += 1

        print(f"  Batches: {n_batches} | Buffer: {len(buf)}")

        # Print lambdas only when they are being meta-learned
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
        evaluate_all(model, task, acc_til, acc_cil, classes_per_task)

    return acc_til, acc_cil
