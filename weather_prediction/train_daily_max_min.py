import torch
import torch.nn as nn
import numpy as np
from read_data import ERA5Dataset

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

if __name__ == "__main__":
    dataset = ERA5Dataset(years=range(2015, 2016), window_size=5, max_or_min='max')
    print(f"Dataset size: {len(dataset)}")
    sample_data, sample_target = dataset[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample target: {sample_target}")
