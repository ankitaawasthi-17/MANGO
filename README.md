# OCL-AMPEWC

Online Continual Learning with Adaptive Meta Parameter EWC.

## Datasets
- CIFAR-100 (20 tasks, 5 classes/task)
- TinyImageNet (20 tasks, 10 classes/task)

## Run

### CIFAR
python3 main_tiny.py --dataset cifar --buffer_size 2000 --seed 42
### TinyImageNet
python3 main_tiny.py --dataset tiny

###MultiBuffer-Multiseed Comands

1. Tinyimagenet buffer 4000: -----------------------DONE!----------------------------


for seed in 42 43 44 45 46; do 
  echo "===== Running TINY | Buffer=4000 | Seed=$seed ====="
  python3 main_tiny.py --dataset tiny --buffer_size 4000 --seed $seed
done

2. CIFAR 100, buffer 4000: ------------------------DONE!


for seed in 42 43 44 45 46; do 
  echo "===== Running CIFAR | Buffer=4000 | Seed=$seed ====="
  python3 main_tiny.py --dataset cifar --buffer_size 4000 --seed $seed
done

3. CIFAR 100, buffer 2000: ------------------------DONE!------------------------------------


for seed in 42 43 44 45 46; do 
  echo "===== Running CIFAR | Buffer=2000 | Seed=$seed ====="
  python3 main_tiny.py --dataset cifar --buffer_size 2000 --seed $seed
done


4. Tinyimagenet buffer 2000: ------------------------ONGOING...--------------------------


for seed in 42 43 44 45 46; do 
  echo "===== Running TINY | Buffer=2000 | Seed=$seed ====="
  python3 main_tiny.py --dataset tiny --buffer_size 2000 --seed $seed
done

5. CIFAR 100, buffer 1000: ------------------------ONGOING...----------------------------


for seed in 42 43 44 45 46; do 
  echo "===== Running CIFAR | Buffer=1000 | Seed=$seed ====="
  python3 main_tiny.py --dataset cifar --buffer_size 1000 --seed $seed
done




--------------------------------------------------------------------------------------------------
BASELINES
--------------------------------------------------------------------------------------------------

1. DER++

DER++ CIFAR-100 baseline BUffer 2000, 4000, 1000, 5 seed

for buffer in 2000 4000 1000; do
  for seed in 42 43 44 45 46; do
    echo "===== DER++ CIFAR | Buffer=$buffer | Seed=$seed ====="
    python3 main_derpp.py --dataset cifar --buffer_size $buffer --seed $seed 
  done
done


DER++ Tiny baseline BUffer 2000, 4000 5 seed

for buffer in 4000 2000; do
  for seed in 42 43 44 45 46; do
    echo "===== DER++ TINY | Buffer=$buffer | Seed=$seed ====="
    python3 main_derpp.py --dataset tiny --buffer_size $buffer --seed $seed --lr 0.05 
  done
done

2. LODE

LODE CIFAR-100 baseline BUffer 2000, 4000, 1000, 5 seed---------ONGOING...--------------

for buffer in 2000 4000 1000; do
  for seed in 42 43 44 45 46; do
    echo "===== LODE CIFAR | Buffer=$buffer | Seed=$seed ====="
    python3 main_lode.py --dataset cifar --buffer_size $buffer --seed $seed
  done
done

LODE Tiny baseline BUffer 2000, 4000 5 seed---------ONGOING...--------------


for buffer in 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== LODE TINY | Buffer=$buffer | Seed=$seed ====="
    python3 main_lode.py --dataset tiny --buffer_size $buffer --seed $seed --lr 0.05
  done
done


3. ER

ER CIFAR-100 baseline BUffer 2000, 4000, 1000, 5 seed---------ONGOING...--------------

for buffer in 1000 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== ER CIFAR | Buffer=$buffer | Seed=$seed ====="
    python3 main_er.py --dataset cifar --buffer_size $buffer --seed $seed
  done
done


ER Tiny baseline BUffer 2000, 4000 5 seed---------ONGOING...--------------

for buffer in 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== ER TINY | Buffer=$buffer | Seed=$seed ====="
    python3 main_er.py --dataset tiny --buffer_size $buffer --seed $seed
  done
done


4. FT


6. iCarL


iCARL CIFAR-100 baseline BUffer 2000, 4000, 1000, 5 seed---------ONGOING...--------------


for buffer in 2000 4000 1000; do
  for seed in 42 43 44 45 46; do
    echo "===== ICARL CIFAR | Buffer=$buffer | Seed=$seed ====="
    python3 main_icarl.py --dataset cifar --buffer_size $buffer --seed $seed
  done
done


iCARL Tiny baseline BUffer 2000, 4000 5 seed---------ONGOING...--------------


for buffer in 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== ICARL TINY | Buffer=$buffer | Seed=$seed ====="
    python3 main_icarl.py --dataset tiny --buffer_size $buffer --seed $seed --lr 0.05
  done
done


7. LUCIR


LUCIR CIFAR-100 baseline BUffer 2000, 4000, 1000, 5 seed---------ONGOING...--------------


for buffer in 1000 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== LUCIR CIFAR | Buffer=$buffer | Seed=$seed ====="
    python3 main_lucir.py --dataset cifar --buffer_size $buffer --seed $seed 
  done
done

LUCIR Tiny baseline BUffer 2000, 4000 5 seed---------ONGOING...--------------


for buffer in 2000 4000; do
  for seed in 42 43 44 45 46; do
    echo "===== LUCIR TINY | Buffer=$buffer | Seed=$seed ====="
    python3 main_lucir.py --dataset tiny --buffer_size $buffer --seed $seed 
  done
done


8. G-Dumb


9. ER-ACE

