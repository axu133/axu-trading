import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import xarray as xr
import numpy as np
from preprocess import z_score_normalize

def central_park_daily_max_min():
    """
    Load daily max/min temperature for Central Park. Returns a DataFrame with columns DATE, TMAX, TMIN.
    """
    file_path = os.path.join("data", "Central_Park_Daily_Max_Min.csv")
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df = df[['DATE', 'TMAX', 'TMIN']]
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

def load_era5(month, year):
    """
    Load ERA5 data for a specific month and year. Trims images to 64x64 and returns 6-hourly data as an xarray DataArray.

    :param month: Month to load (1-12)
    :param year: Year to load (e.g., 2015)
    """
    vars = ['t2m', 'u10', 'v10', 'msl', 'd2m']

    file_path = os.path.join("data", "era5_raw", str(year), f"era5_{year}_{month:02d}.nc") # Path hardcoded
    ds = xr.open_dataset(file_path)

    subset = ds.sel(valid_time = ds.valid_time.dt.hour.isin([0, 6, 12, 18]))

    trimmed = subset.isel(latitude=slice(0,64), longitude=slice(0,64))

    batched = (
        trimmed.coarsen(valid_time=4, boundary='trim')
        .construct(valid_time=('batch','step'))
    )

    data = batched[vars].to_array(dim='channel').transpose('batch', 'step', 'channel', 'latitude', 'longitude')

    time_index = batched['valid_time'].isel(step=0).values

    return data, time_index

def align_dates(era5, labels, target_col = "TMAX", lag_days = 5):
    batch_dates = pd.to_datetime(era5).normalize()

    if batch_dates.tz is not None:
        batch_dates = batch_dates.tz_localize(None)

    target_dates = batch_dates + pd.Timedelta(days=lag_days)

    if 'DATE' in labels.columns:
        labels = labels.set_index('DATE')

    labels.index = pd.to_datetime(labels.index)

    aligned = labels[target_col].reindex(target_dates)

    valid_mask = aligned.notna().values

    final_labels = aligned[valid_mask].values

    final_labels = final_labels.reshape(-1, 1).astype(np.float32)

    return valid_mask, final_labels

class ERA5Dataset(Dataset):
    def __init__(self, years, window_size = 5, max_or_min='max'):
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
        self.data, _, _ = z_score_normalize(data)

        time_index = np.concatenate(time_index, axis=0)

        all_targets = central_park_daily_max_min()

        if max_or_min == 'max':
            target_col = 'TMAX'
        elif max_or_min == 'min':
            target_col = 'TMIN'
        else:
            raise ValueError("max_or_min must be 'max' or 'min'")

        self.window_size = window_size
        self.valid_samples = []

        if 'DATE' in all_targets.columns:
            all_targets = all_targets.set_index('DATE')
        all_targets.index = pd.to_datetime(all_targets.index).normalize()
        if all_targets.index.tz is not None:
            all_targets.index = all_targets.index.tz_localize(None)

        era5_dates = pd.to_datetime(time_index)
        if era5_dates.tz is not None:
            era5_dates.tz = era5_dates.tz_localize(None)
        era5_dates = era5_dates.normalize()

        for i in range(len(self.data) - self.window_size):
            start_date = era5_dates[i]
            end_date = era5_dates[i + self.window_size - 1]

            expected_end = start_date + pd.Timedelta(days=self.window_size - 1)

            if end_date != expected_end:
                continue # Otherwise gaps exist

            target_date = end_date + pd.Timedelta(days=1)
            if target_date in all_targets.index:
                target_val = all_targets.loc[target_date, target_col]
                if not pd.isna(target_val):
                    self.valid_samples.append((i, target_val))
            
        if len(self.valid_samples) == 0:
            raise ValueError("No valid samples found. Check your data and date alignment.")

    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        start_idx, target_val = self.valid_samples[idx]

        seq = self.data[start_idx: start_idx + self.window_size]

        data = torch.tensor(seq, dtype=torch.float32)
        target = torch.tensor(target_val, dtype=torch.float32)
        return data, target


if __name__ == "__main__":
    #data = central_park_daily_max_min()
    #print(data["DATE"][0])
    data, _ = load_era5(6, 2015)
    print(data)