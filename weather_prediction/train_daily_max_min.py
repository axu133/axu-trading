import torch
from read_data import central_park_daily_max_min, load_era5

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
