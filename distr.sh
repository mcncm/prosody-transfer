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
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node742 --exclusive task0.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node744 --exclusive task1.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node745 --exclusive task2.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node746 --exclusive task3.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node747 --exclusive task4.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node748 --exclusive task5.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node749 --exclusive task6.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node750 --exclusive task7.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node751 --exclusive task8.sh &
srun -l --nodes=1 --ntasks=1 --gres=gpu:2 --nodelist=node752 --exclusive task9.sh &
wait

