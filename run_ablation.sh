#!/bin/bash
# run_ablation.sh
# Runs all four methods across 3 seeds, then aggregates results.
#
# Usage:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh                    # TinyImageNet (default)
#   ./run_ablation.sh --dataset cifar    # CIFAR-100

DATASET=${1:-tiny}
SEEDS=(42 43 44)
RESULTS_DIR="./results_ablation"
METHODS=("ours" "fixed_lambda" "no_reg" "er")

echo "======================================================="
echo "  OCL Ablation Study — Dataset: $DATASET"
echo "  Methods: ${METHODS[*]}"
echo "  Seeds  : ${SEEDS[*]}"
echo "======================================================="

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo "  Method=$METHOD | Seed=$SEED"
        echo "----------------------------------------------"
        python3 main_tiny.py \
            --method $METHOD \
            --dataset $DATASET \
            --seed $SEED \
            --results_dir $RESULTS_DIR
    done
done

echo ""
echo "======================================================="
echo "  Aggregating results..."
echo "======================================================="

python3 - << EOF
import numpy as np, os, glob

results_dir = "./results_ablation"
dataset     = "$DATASET"
methods     = ["ours", "fixed_lambda", "no_reg", "er"]

def load_metrics(mat_path):
    m   = np.load(mat_path)
    n   = m.shape[0]
    acc = np.mean(m[n-1, :n])
    bwt = np.mean([m[n-1, t] - m[t, t] for t in range(n-1)])
    aaa = np.mean([np.mean(m[t, :t+1])  for t in range(n)])
    wc  = np.mean([np.min(m[t,  :t+1])  for t in range(n)])
    return acc, bwt, aaa, wc

header = f"{'Method':<14} {'CIL-ACC':>10} {'TIL-ACC':>10} {'CIL-BWT':>10} {'WC-Acc':>10} {'AAA':>10}"
print()
print(header)
print("─" * len(header))

for method in methods:
    cil_accs, til_accs, cil_bwts, wcs, aaas = [], [], [], [], []
    for seed_dir in sorted(glob.glob(
            os.path.join(results_dir, dataset, method, "seed_*"))):
        cil_path = os.path.join(seed_dir, "acc_matrix_cil.npy")
        til_path = os.path.join(seed_dir, "acc_matrix_til.npy")
        if not (os.path.exists(cil_path) and os.path.exists(til_path)):
            continue
        ca, cb, caaa, cwc = load_metrics(cil_path)
        ta, tb, taaa, twc = load_metrics(til_path)
        cil_accs.append(ca); cil_bwts.append(cb)
        til_accs.append(ta); aaas.append(caaa); wcs.append(cwc)

    def fmt(vals):
        if not vals: return "  N/A"
        return f"{np.mean(vals):6.2f}±{np.std(vals):.2f}"

    print(f"{method:<14} {fmt(cil_accs):>10} {fmt(til_accs):>10} "
          f"{fmt(cil_bwts):>10} {fmt(wcs):>10} {fmt(aaas):>10}")

if dataset == "tiny":
    print()
    print("DER++ targets: CIL=11.61 | TIL=48.87 | BWT=-34.6 | WC=10.35 | AAA=17.85")
EOF

echo "Done."
