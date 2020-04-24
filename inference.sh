#!/bin/bash
#
#SBATCH --job-name=infer
#SBATCH --chdir=/home/szha0/tacotron2
#SBATCH --output=output/%x-%J.out
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#
#SBATCH --partition=mit-6345
#SBATCH --time=2-00:00:00
#SBATCH --mem=125G
#
#SBATCH --nodelist=node743

hostname
source /home/szha0/anaconda2/etc/profile.d/conda.sh
conda activate test7
source /home/szha0/.matplotlib/matplotlibrc
python inference.py

