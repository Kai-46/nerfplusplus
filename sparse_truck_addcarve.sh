#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c 10
####SBATCH -C turing
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%A.out


PYTHON=/home/zhangka2/anaconda3/envs/nerf-ddp/bin/python

CODE_DIR=/home/zhangka2/gernot_experi/nerf_bg_latest_ddp
echo $CODE_DIR

$PYTHON -u $CODE_DIR/ddp_run_nerf.py --config $CODE_DIR/configs/tanks_and_temples_sparse/tat_training_truck_addcarve.txt
