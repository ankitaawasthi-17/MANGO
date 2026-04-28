import argparse
import torch
from models.er import ER
from models.ft import FT
from models.er_ace import ER_ACE
from models.gdumb import GDumb
from models.derpp import DERPP
from models.lode import LODE
from utils.seed import set_seed
from models.ocl_resnet18_tiny import OCLResNet18Tiny
from train.trainer_clear import train_sequence, compute_final_metrics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(name, buffer_size):
    if name == "mango":
        return OCLResNet18Tiny(num_classes=10, lr=0.05, dataset="clear")
    elif name == "er":
        return ER(num_classes=10, buffer_size=buffer_size)
    elif name == "ft":
        return FT(num_classes=10)
    elif name == "er_ace":
        return ER_ACE(num_classes=10, buffer_size=buffer_size)
    elif name == "gdumb":
        return GDumb(num_classes=10, buffer_size=buffer_size)
    elif name == "derpp":
        return DERPP(num_classes=10, buffer_size=buffer_size)
    elif name == "lode":
        return LODE(num_classes=10, buffer_size=buffer_size)
    else:
        raise ValueError("Unknown model")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--model",
    type=str,
    default="mango",
    choices=["mango", "er", "ft", "er_ace", "gdumb", "derpp", "lode"]
)
    parser.add_argument("--buffer_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    num_tasks = 10
    classes_per_task = 10

    print("\n===== DATASET: CLEAR =====")

    model = get_model(args.model, args.buffer_size).to(DEVICE)

    acc_til, acc_cil = train_sequence(
        model=model,
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        glances=3,
        epochs=1,
        buffer_size=args.buffer_size,
    )

    print("\n===== FINAL =====")
    compute_final_metrics(acc_til, "TIL")
    compute_final_metrics(acc_cil, "CIL")


if __name__ == "__main__":
    main()
