#!/bin/bash

#SBATCH --job-name=WeatherPred
#SBATCH --output=Train.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python weather_prediction/train_daily_max_min.py