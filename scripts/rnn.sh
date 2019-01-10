#!/bin/bash
#SBATCH -J rnn
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/rnn.yaml
srun python train.py $config 