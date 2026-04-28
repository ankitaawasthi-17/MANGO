import numpy as np
import torch

from data.task_split_clear import get_task_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_sequence(model, num_tasks, classes_per_task, glances, epochs, buffer_size):
    acc_til = np.zeros((num_tasks, num_tasks))
    acc_cil = np.zeros((num_tasks, num_tasks))

    for task in range(num_tasks):

        print("\n" + "="*55)
        print(f"  TASK {task}")
        print("="*55)

        loader = get_task_loader(task, classes_per_task, train=True, batch_size=32)

        # -------------------------
        # TRAIN
        # -------------------------
        for epoch in range(epochs):
            for _ in range(glances):
                for x, y in loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    model.observe(x, y)

        # -------------------------
        # EVAL
        # -------------------------
        evaluate(model, task, acc_til, acc_cil, classes_per_task)

    return acc_til, acc_cil


def evaluate(model, task, acc_til, acc_cil, cpt):
    model.eval()

    with torch.no_grad():
        for t in range(task + 1):
            loader = get_task_loader(t, cpt, train=False, batch_size=128)

            correct_til = 0
            correct_cil = 0
            total = 0

            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)

                # 🔥 CLEAR = no class expansion → TIL == CIL
                preds = torch.argmax(logits, dim=1)

                correct_til += (preds == y).sum().item()
                correct_cil += (preds == y).sum().item()
                total += y.size(0)

            acc_til[task, t] = 100 * correct_til / total
            acc_cil[task, t] = 100 * correct_cil / total

    print("\n--- Evaluation ---")
    print("TIL:", np.round(acc_til[task, :task+1], 1))
    print("CIL:", np.round(acc_cil[task, :task+1], 1))


def compute_final_metrics(acc_matrix, name):
    final_acc = acc_matrix[-1]
    avg_acc = np.mean(final_acc)

    print(f"\n{name} FINAL ACC: {avg_acc:.2f}")
