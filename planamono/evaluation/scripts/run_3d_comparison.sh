#!/bin/bash
#SBATCH --job-name=3d_vis
#SBATCH --output=/cluster/scratch/ayavuz/logs/3d_vis_%j.out
#SBATCH --error=/cluster/scratch/ayavuz/logs/3d_vis_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1

source ~/.bashrc
conda activate moge

cd /cluster/home/ayavuz/PixelwisePlanarity

python planamono/evaluation/qualitative/generate_3d_comparison.py \
    --scene_list planamono/evaluation/qualitative/scannetpp_vis_scenes.txt \
    --output_root /cluster/scratch/ayavuz/3d_vis/scannetpp \
    --checkpoint /cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt \
    --num_tokens 1600 \
    --rotations 0 1 2 3 \
    --rot_x 20 \
    --point_radius 1
