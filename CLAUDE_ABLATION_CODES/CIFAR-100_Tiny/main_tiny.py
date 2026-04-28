# main_tiny.py

import argparse, os
import numpy as np
import torch

from utils.seed import set_seed
from models.ocl_resnet18_tiny import OCLResNet18Tiny, VALID_METHODS
from train.trainer_tinyimagenet import train_sequence, compute_final_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "tiny":  {"num_tasks": 20, "classes_per_task": 10,
               "glances": 1,  "default_lr": 0.05},
    "cifar": {"num_tasks": 20, "classes_per_task":  5,
               "glances": 3,  "default_lr": 0.02},
}

# ── DER++ targets for comparison table ───────────────────────────────────────
DERPP_TARGETS = {
    "tiny":  {"CIL": 11.61, "TIL": 48.87, "BWT": -34.6,
               "WC":  10.35, "AAA": 17.85},
    "cifar": {"CIL": None,  "TIL": None,  "BWT": None,
               "WC":  None,  "AAA": None},
}


def main():
    parser = argparse.ArgumentParser(
        description="OCL Ablation Study",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--method", type=str, default="ours",
        choices=VALID_METHODS,
        help=(
            "ours         → Full method: meta-learned lambda + Amphibian\n"
            "fixed_lambda → Ablation: fixed lambda=0.001 + Amphibian (no meta-learning)\n"
            "no_reg       → Ablation: no regularization + Amphibian\n"
            "er           → Baseline: plain Experience Replay (no lambda, no gating)"
        ),
    )
    parser.add_argument("--dataset",     type=str,   default="tiny",
                        choices=["tiny", "cifar"])
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--results_dir", type=str,   default="./results_ablation")
    parser.add_argument("--lr",          type=float, default=None,
                        help="Override LR (default: 0.05 tiny / 0.02 cifar)")
    parser.add_argument("--buffer_size", type=int,   default=None,
                        help="Override buffer (default: 4000 tiny / 2000 cifar)")
    parser.add_argument("--data_root",   type=str,
                        default="./data/tiny-imagenet-200")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    lr  = args.lr if args.lr is not None else cfg["default_lr"]
    lambda_lr = 2e-3 if args.dataset == "cifar" else 1e-3

    set_seed(args.seed)

    seed_dir = os.path.join(
        args.results_dir, args.dataset, args.method, f"seed_{args.seed}"
    )
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  OCL Ablation Study")
    print(f"{'='*55}")
    print(f"  Method  : {args.method}")
    print(f"  Dataset : {args.dataset}  |  Seed: {args.seed}")
    print(f"  LR      : {lr}  |  lambda_lr: {lambda_lr}")
    print(f"  Tasks   : {cfg['num_tasks']} × {cfg['classes_per_task']} classes")
    print(f"  Glances : {cfg['glances']}")
    print(f"{'='*55}\n")

    if args.dataset == "tiny":
        import data.task_split_tinyimagenet as tsplit
        tsplit._DATA_ROOT = args.data_root

    model = OCLResNet18Tiny(
        num_classes=cfg["num_tasks"] * cfg["classes_per_task"],
        lr=lr,
        dataset=args.dataset,
        lambda_lr=lambda_lr,
        method=args.method,
    ).to(DEVICE)

    print(f"  Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    acc_til, acc_cil = train_sequence(
        model=model,
        num_tasks=cfg["num_tasks"],
        classes_per_task=cfg["classes_per_task"],
        glances=cfg["glances"],
        epochs=1,
        buffer_size=args.buffer_size,
    )

    np.save(os.path.join(seed_dir, "acc_matrix_til.npy"), acc_til)
    np.save(os.path.join(seed_dir, "acc_matrix_cil.npy"), acc_cil)

    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS — {args.method.upper()} on {args.dataset.upper()}")
    print(f"{'='*55}")
    til = compute_final_metrics(acc_til, "TIL")
    cil = compute_final_metrics(acc_cil, "CIL")

    # Comparison table (only meaningful for TinyImageNet vs DER++)
    if args.dataset == "tiny":
        t = DERPP_TARGETS["tiny"]
        print(f"\n  {'Metric':<12} {'DER++':<12} {'Ours/Ablation'}")
        print(f"  {'─'*38}")
        print(f"  {'CIL Acc':<12} {t['CIL']:<12} {cil['acc']:.2f}")
        print(f"  {'TIL Acc':<12} {t['TIL']:<12} {til['acc']:.2f}")
        print(f"  {'CIL BWT':<12} {t['BWT']:<12} {cil['bwt']:.2f}")
        print(f"  {'WC-Acc':<12} {t['WC']:<12}  {cil['wc_acc']:.2f}")
        print(f"  {'AAA':<12} {t['AAA']:<12} {cil['aaa']:.2f}")

    print(f"\n  Results saved: {seed_dir}")


if __name__ == "__main__":
    main()
