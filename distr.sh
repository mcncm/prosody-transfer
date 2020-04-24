#!/bin/bash
#
#SBATCH --job-name=distr
#SBATCH --chdir=/home/szha0/tacotron2
#SBATCH --output=output/%x-%J.out
#
#SBATCH --nodes=10
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#
#SBATCH --partition=mit-6345
#SBATCH --time=2-00:00:00
#SBATCH --mem=125G

hostname
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node742 task0.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node744 task1.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node745 task2.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node746 task3.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node747 task4.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node748 task5.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node749 task6.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node750 task7.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node751 task8.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node752 task9.sh &
wait

