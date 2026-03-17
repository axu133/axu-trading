#!/bin/bash

#SBATCH --job-name=TMAX_GBDT
#SBATCH --output=baseline_train.txt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

# Gradient-boosted tree baseline for next-day Central Park TMAX (°F)
# Uses time-based splits and train-only normalization inside the script.
python -u -m weather_prediction.train_tabular_tmax

