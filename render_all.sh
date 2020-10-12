#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c 10
#SBATCH -C pascal
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%A.out
#SBATCH --qos=high

PYTHON=/home/zhangka2/anaconda3/envs/nerf-ddp/bin/python

CODE_DIR=/home/zhangka2/gernot_experi/nerf_bg_latest_ddp
echo $CODE_DIR

#$PYTHON -u $CODE_DIR/ddp_test_nerf.py --config $CODE_DIR/configs/lf_data/lf_africa.txt


$PYTHON -u $CODE_DIR/ddp_test_nerf.py --config $CODE_DIR/configs/tanks_and_temples/tat_training_truck.txt
