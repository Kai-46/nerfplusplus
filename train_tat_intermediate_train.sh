#!/bin/bash
#SBATCH -p q6
#SBATCH --gres=gpu:3
#SBATCH -c 8
#SBATCH -C turing
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%A.out


PYTHON=/home/zhangka2/anaconda3/envs/nerf/bin/python

CODE_DIR=/home/zhangka2/gernot_experi/nerf_bg_latest
echo $CODE_DIR

$PYTHON -u $CODE_DIR/run_nerf.py --config $CODE_DIR/configs/tanks_and_temples/tat_intermediate_train.txt
$PYTHON -u $CODE_DIR/nerf_render_image.py --config $CODE_DIR/configs/tanks_and_temples/tat_intermediate_train.txt

