# main_tiny.py

import argparse, os
import numpy as np
import torch

from utils.seed import set_seed
from models.ocl_resnet18_tiny import OCLResNet18Tiny
from train.trainer_tinyimagenet import train_sequence, compute_final_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 DEFAULT = Tiny (unchanged)
NUM_TASKS        = 10
CLASSES_PER_TASK = 20
GLANCES          = 3
EPOCHS_PER_TASK  = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--dataset", choices=["tiny","cifar"], default="tiny")  # ✅ NEW
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--results_dir", type=str,   default="./results_tiny")
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--no_meta",     action="store_true")
    parser.add_argument("--data_root",   type=str, default="./data/tiny-imagenet-200")
    args = parser.parse_args()

    set_seed(args.seed)

    # 🔥 CIFAR CONFIG (does NOT affect Tiny)
    if args.dataset == "cifar":
        num_tasks = 20
        classes_per_task = 5
        lr = 0.02 if args.lr == 0.05 else args.lr
    else:
        num_tasks = NUM_TASKS
        classes_per_task = CLASSES_PER_TASK
        lr = args.lr

    seed_dir = os.path.join(args.results_dir, f"{args.dataset}_seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n===== DATASET: {args.dataset.upper()} =====")

    # Tiny root override (unchanged)
    if args.dataset == "tiny":
        import data.task_split_tinyimagenet as tsplit
        tsplit._DATA_ROOT = args.data_root

    model = OCLResNet18Tiny(
        num_classes=num_tasks * classes_per_task,
        lr=lr,
        no_meta=args.no_meta,
        dataset=args.dataset,   # ✅ NEW
        lambda_lr = 2e-3 if args.dataset=="cifar" else 1e-3
    ).to(DEVICE)

    acc_til, acc_cil = train_sequence(
        model=model,
    num_tasks=num_tasks,
    classes_per_task=classes_per_task,
    glances=GLANCES,
    epochs=EPOCHS_PER_TASK,
    buffer_size=args.buffer_size,
    )

    print("\n===== FINAL =====")
    compute_final_metrics(acc_til, "TIL")
    compute_final_metrics(acc_cil, "CIL")


if __name__ == "__main__":
    main()
