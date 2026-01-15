import torch
from torch.utils.data import Dataset
import numpy as np
from preprocess import z_score_normalize, align_dates
from read_data import central_park_daily_max_min, load_era5

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class ERA5Dataset(Dataset):
    def __init__(self, years, max_or_min='max'):
        """
        ERA5 Dataset for loading ERA5 weather data and corresponding Central Park daily max/min temperatures.
        :param years: List of years to include in the dataset. Will start on Jan 1 and end on Dec 31 of the specified years.
        :param max_or_min: 'max' for daily max temperatures, 'min' for daily min temperatures.
        """
        data = []
        time_index = []

        for year in years:
            for month in range(1, 13):
                era5_data, era5_time_index = load_era5(month, year)
                data.append(era5_data.values)
                time_index.append(era5_time_index)

        data = np.concatenate(data, axis=0)
        time_index = np.concatenate(time_index, axis=0)

        all_targets = central_park_daily_max_min()

        if max_or_min == 'max':
            target_col = 'TMAX'
        elif max_or_min == 'min':
            target_col = 'TMIN'
        else:
            raise ValueError("max_or_min must be 'max' or 'min'")

        mask, targets = align_dates(
            time_index,
            all_targets,
            target_col=target_col,
            lag_days=5
        )

        self.data, _, _ = z_score_normalize(data[mask])
        self.targets = targets

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return data, target



