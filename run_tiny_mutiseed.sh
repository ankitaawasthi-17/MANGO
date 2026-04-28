#!/bin/bash
# run_tiny_multiseed.sh
# Runs TinyImageNet OCL experiments across 3 seeds and prints aggregated results.
#
# Usage:
#   chmod +x run_tiny_multiseed.sh
#   ./run_tiny_multiseed.sh
#
# Optional: pass --no_meta to run fixed-lambda ablation
# Optional: pass --lr 0.05 to override learning rate

SEEDS=(42 43 44)
RESULTS_DIR="./results_tiny"
DATA_ROOT="./data/tiny-imagenet-200"
LR=0.1
EXTRA_ARGS="$@"   # forward any extra flags (e.g. --no_meta)

echo "======================================================="
echo "  TinyImageNet OCL — Multi-Seed Run"
echo "  Seeds: ${SEEDS[*]}"
echo "  LR: $LR | Extra: $EXTRA_ARGS"
echo "======================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------"
    echo "  Seed: $SEED"
    echo "------------------------------------"
    python3 main_tiny.py \
        --seed $SEED \
        --lr $LR \
        --results_dir $RESULTS_DIR \
        --data_root $DATA_ROOT \
        $EXTRA_ARGS
done

echo ""
echo "======================================================="
echo "  Aggregating results across seeds..."
echo "======================================================="

python3 - <<'EOF'
import numpy as np, os, glob

results_dir = "./results_tiny"
til_accs, cil_accs = [], []
til_bwts, cil_bwts = [], []
til_aaas, cil_aaas = [], []
til_wcs,  cil_wcs  = [], []

for seed_dir in sorted(glob.glob(os.path.join(results_dir, "seed_*"))):
    til_path = os.path.join(seed_dir, "acc_matrix_til.npy")
    cil_path = os.path.join(seed_dir, "acc_matrix_cil.npy")
    if not (os.path.exists(til_path) and os.path.exists(cil_path)):
        continue

    for path, accs, bwts, aaas, wcs in [
        (til_path, til_accs, til_bwts, til_aaas, til_wcs),
        (cil_path, cil_accs, cil_bwts, cil_aaas, cil_wcs),
    ]:
        m = np.load(path)
        n = m.shape[0]
        acc  = np.mean(m[n-1, :n])
        bwt  = np.mean([m[n-1, t] - m[t, t] for t in range(n-1)])
        aaa  = np.mean([np.mean(m[t, :t+1]) for t in range(n)])
        wc   = np.mean([np.min(m[t, :t+1]) for t in range(n)])
        accs.append(acc); bwts.append(bwt); aaas.append(aaa); wcs.append(wc)

def fmt(vals):
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

print(f"\n{'Setting':<8} {'ACC':>16} {'AAA':>16} {'WC-Acc':>16} {'BWT':>16}")
print(f"{'─'*76}")
print(f"{'TIL':<8} {fmt(til_accs):>16} {fmt(til_aaas):>16} {fmt(til_wcs):>16} {fmt(til_bwts):>16}")
print(f"{'CIL':<8} {fmt(cil_accs):>16} {fmt(cil_aaas):>16} {fmt(cil_wcs):>16} {fmt(cil_bwts):>16}")
print()
print("DER++ targets: CIL=11.61 | TIL=48.87 | WC-Acc=10.35 | AAA=17.85 | BWT=-34.6")
EOF

echo "All done."
