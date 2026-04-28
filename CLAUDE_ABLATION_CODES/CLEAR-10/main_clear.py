# main_clear.py

import argparse
import os
import numpy as np
import torch

from utils.seed import set_seed
from models.ocl_resnet18_tiny import OCLResNet18Tiny, VALID_METHODS
from train.trainer_clear import train_sequence, compute_final_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLEAR-10 config (fixed — not a hyperparameter)
NUM_TASKS        = 10
CLASSES_PER_TASK = 10   # all 10 classes present from task 0
GLANCES          = 3
EPOCHS_PER_TASK  = 1


def main():
    parser = argparse.ArgumentParser(
        description="CLEAR-10 OCL Ablation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--method", type=str, default="ours",
        choices=VALID_METHODS,
        help=(
            "ours         → Meta-learned lambda + Amphibian\n"
            "fixed_lambda → Fixed lambda=0.001 + Amphibian (no meta-learning)\n"
            "no_reg       → No regularization + Amphibian\n"
            "er           → Plain Experience Replay (no lambda, no gating)"
        ),
    )
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=2000)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--results_dir", type=str, default="./results_clear")
    args = parser.parse_args()

    set_seed(args.seed)

    seed_dir = os.path.join(
        args.results_dir, args.method,
        f"buf{args.buffer_size}", f"seed_{args.seed}"
    )
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CLEAR-10 OCL Ablation")
    print(f"{'='*55}")
    print(f"  Method      : {args.method}")
    print(f"  Seed        : {args.seed}")
    print(f"  Buffer      : {args.buffer_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Tasks       : {NUM_TASKS} × {CLASSES_PER_TASK} classes")
    print(f"  Glances     : {GLANCES}")
    print(f"{'='*55}\n")

    model = OCLResNet18Tiny(
        num_classes=CLASSES_PER_TASK,   # 10 — no expansion on CLEAR
        lr=args.lr,
        dataset="clear",
        lambda_lr=1e-3,
        method=args.method,
    ).to(DEVICE)

    print(f"  Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    acc_til, acc_cil = train_sequence(
        model=model,
        num_tasks=NUM_TASKS,
        classes_per_task=CLASSES_PER_TASK,
        glances=GLANCES,
        epochs=EPOCHS_PER_TASK,
        buffer_size=args.buffer_size,
    )

    np.save(os.path.join(seed_dir, "acc_matrix_til.npy"), acc_til)
    np.save(os.path.join(seed_dir, "acc_matrix_cil.npy"), acc_cil)

    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS — {args.method.upper()}")
    print(f"{'='*55}")
    compute_final_metrics(acc_til, "TIL")
    compute_final_metrics(acc_cil, "CIL")

    print(f"\n  Results saved: {seed_dir}")


if __name__ == "__main__":
    main()
