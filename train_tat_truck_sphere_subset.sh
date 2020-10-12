#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH -c 8
#SBATCH -C turing
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%A.out
#SBATCH --exclude=isl-gpu17


PYTHON=/home/zhangka2/anaconda3/envs/nerf/bin/python

CODE_DIR=/home/zhangka2/gernot_experi/nerf_bg
echo $CODE_DIR

$PYTHON $CODE_DIR/run_nerf.py --config $CODE_DIR/configs/tanks_and_temples/tat_training_truck_subset.txt
$PYTHON $CODE_DIR/nerf_render_image.py --config $CODE_DIR/configs/tanks_and_temples/tat_training_truck_subset.txt

