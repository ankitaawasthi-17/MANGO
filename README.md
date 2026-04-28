# 🥭 MANGO: Memory-Augmented Online Continual Learning with Adaptive Regularization

## Overview
This repository contains the official implementation of MANGO, an Online Continual Learning (OCL) method that combines experience replay with adaptive regularization to improve stability and mitigate catastrophic forgetting.

We evaluate MANGO across multiple challenging continual learning benchmarks:

- CIFAR-100 (Split-100)
- TinyImageNet (Split-200)
- CLEAR-10 (Temporal Continual Learning)

The repository also includes strong baselines:
- ER (Experience Replay)
- DER++
- LODE
- ER-ACE
- GDumb
- iCaRL
- LUCIR
- Fine-Tuning (FT)

---

# ⚙️ Installation

bash git clone https://github.com/ankitaawasthi-17/MANGO.git cd MANGO  python3 -m venv venv source venv/bin/activate  pip install torch torchvision numpy pandas scikit-learn 

---

# 📦 Datasets

## CIFAR-100
- Automatically downloaded via torchvision

---

## TinyImageNet

Download and place in:

text data/tiny-imagenet-200/ 

---

## 🧠 CLEAR-10 (Key Benchmark)

CLEAR-10 is a temporal continual learning dataset, where data is split into 10 sequential time steps instead of random class splits.

### 📁 Expected Structure

text data/CLEAR/   train_image_only/     class_names.txt     labeled_images/       1/laptop/*.jpg       1/camera/*.jpg       ...     labeled_metadata/       1/laptop.json       1/camera.json       ... 

Each folder 1–10 corresponds to a time step, introducing real-world distribution shifts.

---

# 🚀 Running Experiments

## 🔹 Entry Points

| File | Dataset |
|------|--------|
| main.py | CIFAR-100 |
| main_tiny.py | TinyImageNet + CIFAR |
| main_clear.py | CLEAR-10 |

---

# 🥭 MANGO Experiments

## CIFAR-100

bash for buffer in 1000 2000 4000; do   for seed in 42 43 44 45 46; do     echo "===== CIFAR | Buffer=$buffer | Seed=$seed ====="     python3 main_tiny.py --dataset cifar --buffer_size $buffer --seed $seed   done done 

---

## TinyImageNet

bash for buffer in 2000 4000; do   for seed in 42 43 44 45 46; do     echo "===== TINY | Buffer=$buffer | Seed=$seed ====="     python3 main_tiny.py --dataset tiny --buffer_size $buffer --seed $seed   done done 

---

## 🔥 CLEAR-10 (Primary Contribution)

bash for buffer in 2000 4000; do   for seed in 42 43 44 45 46; do     echo "===== CLEAR | Buffer=$buffer | Seed=$seed ====="     python3 main_clear.py --buffer_size $buffer --seed $seed   done done 

---

# 📊 Baselines

## Example: ER (CIFAR-100)

bash for buffer in 1000 2000 4000; do   for seed in 42 43 44 45 46; do     python3 main_er.py --dataset cifar --buffer_size $buffer --seed $seed   done done 

---

## CLEAR Baselines

bash for model in er ft er_ace gdumb derpp lode; do   for buffer in 2000 4000; do     for seed in 42 43 44 45 46; do       echo "===== RUNNING: $model | CLEAR | Buffer=$buffer | Seed=$seed ====="       python3 main_clear.py --model $model --buffer_size $buffer --seed $seed     done   done done 

---

# 🔬 Ablation Study

We evaluate the contribution of each component:

- MANGO (full model)
- Fixed λ (no adaptation)
- No Regularization
- Pure Replay (ER)

## Run Full Ablation

bash bash run_multiseed_multibuffer.sh 

---

## CLEAR Ablation

bash bash run_clear_ablation.sh 

---

# 📈 Evaluation Metrics

We report:

- CIL Accuracy (Class Incremental)
- TIL Accuracy (Task Incremental)
- BWT (Backward Transfer)
- AAA (Average Accuracy)
- Worst-Case Accuracy

---

# 🧠 Implementation Details

- Backbone: ResNet-18
- Training: Online (1 epoch per task)
- Replay buffer sizes: 1000 / 2000 / 4000
- Seeds: 42–46

CLEAR-specific:
- task_split_clear.py → temporal data loading  
- trainer_clear.py → training loop  

---

# 📁 Repository Structure

text . ├── data/ ├── models/ ├── train/ ├── Baselines/ ├── main.py ├── main_tiny.py ├── main_clear.py ├── run_ablation.sh ├── run_multiseed_multibuffer.sh └── Results_*.txt 

---

# ⚠️ Notes

- GPU recommended for faster training
- Do NOT commit:
text data/ venv/ 

---

# 📌 Reproducibility

All experiments use:

- Seeds: 42–46
- Buffers: 1000 / 2000 / 4000
- Single epoch online training

---

# 📄 Citation

bibtex @article{mango2026,   title={MANGO: Memory-Augmented Online Continual Learning with Adaptive Regularization},   author={Awasthi, Ankita},   year={2026} } 

---

# 🙏 Acknowledgements

This work builds on prior continual learning methods including ER, DER++, LODE, and others.

--
