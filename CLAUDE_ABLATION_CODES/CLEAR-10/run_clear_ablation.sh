#!/bin/bash
# run_clear_ablation.sh
# Multi-seed multi-buffer ablation on CLEAR-10.
#
# Usage:
#   chmod +x run_clear_ablation.sh
#   ./run_clear_ablation.sh

SEEDS=(42 43 44 45 46)
METHODS=("ours" "fixed_lambda" "no_reg" "er")
BUFFERS=(1000 2000)
RESULTS_DIR="./results_clear"

echo "======================================================="
echo "  CLEAR-10 Ablation | Methods: ${METHODS[*]}"
echo "  Seeds: ${SEEDS[*]} | Buffers: ${BUFFERS[*]}"
echo "======================================================="

for BUFFER in "${BUFFERS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "  Method=$METHOD | Buffer=$BUFFER | Seed=$SEED"
            python3 main_clear.py \
                --method      $METHOD \
                --seed        $SEED \
                --buffer_size $BUFFER \
                --results_dir $RESULTS_DIR
        done
    done
done

echo ""
echo "======================================================="
echo "  Aggregating results..."
echo "======================================================="

python3 - << 'EOF'
import numpy as np, os, glob

methods = ["ours", "fixed_lambda", "no_reg", "er"]
buffers = [1000, 2000]
results_dir = "./results_clear"

def load_metrics(path):
    m   = np.load(path)
    n   = m.shape[0]
    acc = np.mean(m[n-1, :n])
    bwt = np.mean([m[n-1, t] - m[t, t] for t in range(n-1)])
    aaa = np.mean([np.mean(m[t, :t+1])  for t in range(n)])
    wc  = np.mean([np.min(m[t,  :t+1])  for t in range(n)])
    return acc, bwt, aaa, wc

def fmt(vals):
    if not vals: return "   N/A    "
    return f"{np.mean(vals):5.2f}±{np.std(vals):.2f}"

for buf in buffers:
    print(f"\n{'='*72}")
    print(f"  CLEAR-10 | Buffer={buf}")
    print(f"{'='*72}")
    hdr = f"  {'Method':<14} {'ACC':>12} {'BWT':>12} {'AAA':>12} {'WC-Acc':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for method in methods:
        accs, bwts, aaas, wcs = [], [], [], []
        pattern = os.path.join(
            results_dir, method, f"buf{buf}", "seed_*"
        )
        for seed_dir in sorted(glob.glob(pattern)):
            cp = os.path.join(seed_dir, "acc_matrix_cil.npy")
            if not os.path.exists(cp):
                continue
            a, b, aa, wc = load_metrics(cp)
            accs.append(a); bwts.append(b)
            aaas.append(aa); wcs.append(wc)

        print(f"  {method:<14} {fmt(accs):>12} {fmt(bwts):>12} "
              f"{fmt(aaas):>12} {fmt(wcs):>12}")

print("\nDone.")
EOF

echo "All runs complete."
