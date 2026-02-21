import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import xarray as xr
import numpy as np
from preprocess import z_score_normalize

hours = range(0, 24) #[0, 6, 12, 18]
nhours = len(hours)

def central_park_daily_max_min():
    """
    Load daily max/min temperature for Central Park. Returns a DataFrame with columns DATE, TMAX, TMIN.
    """
    file_path = os.path.join("data", "Central_Park_Daily_Max_Min.csv")
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df = df[['DATE', 'TMAX', 'TMIN']]
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

# Core variables used by the model; optional vars (ssrd, tcc) may exist in clean_data files.
ERA5_CORE_VARS = ['t2m', 'u10', 'v10', 'msl', 'd2m']
ERA5_OPTIONAL_VARS = ['ssrd', 'tcc']  # from combined .grib; use if present for extra channels


def load_era5(month, year, data_root="data", use_optional_vars=True):
    """
    Load ERA5 data for a specific month and year. Trims to 64x64, returns (data, time_index).
    Prefers data/clean_data/era5_YYYY.nc when present (one file per year); else data/era5_raw/YYYY/era5_YYYY_MM.nc.

    :param month: Month to load (1-12)
    :param year: Year to load (e.g., 2015)
    :param data_root: Root data directory (default "data")
    :param use_optional_vars: If True (default), include ssrd/tcc when present (7 channels); else core only (5).
    """
    clean_path = os.path.join(data_root, "clean_data", f"era5_{year}.nc")
    raw_path = os.path.join(data_root, "era5_raw", str(year), f"era5_{year}_{month:02d}.nc")

    if os.path.isfile(clean_path):
        ds = xr.open_dataset(clean_path)
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        if time_dim == "time":
            ds = ds.rename({"time": "valid_time"})
        ds = ds.sel(valid_time=ds.valid_time.dt.month == month)
    else:
        ds = xr.open_dataset(raw_path)

    subset = ds
    trimmed = subset.isel(latitude=slice(0, 64), longitude=slice(0, 64))

    batched = (
        trimmed.coarsen(valid_time=nhours, boundary='trim')
        .construct(valid_time=('batch', 'step'))
    )

    vars_to_use = list(ERA5_CORE_VARS)
    if use_optional_vars:
        vars_to_use += [v for v in ERA5_OPTIONAL_VARS if v in batched.data_vars]
    vars_to_use = [v for v in vars_to_use if v in batched.data_vars]
    if not vars_to_use:
        raise ValueError(f"No expected variables found in data (need at least one of {ERA5_CORE_VARS})")

    data = batched[vars_to_use].to_array(dim='channel').transpose('batch', 'step', 'channel', 'latitude', 'longitude')
    time_index = batched['valid_time'].isel(step=0).values
    ds.close()
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
    def __init__(self, years, window_size=5, max_or_min='max', data_root="data", use_optional_vars=True):
        """
        ERA5 Dataset for loading ERA5 weather data and corresponding Central Park daily max/min temperatures.
        Uses clean_data/era5_YYYY.nc when present (combined files, may include ssrd/tcc).

        :param years: List of years to include in the dataset.
        :param window_size: Number of consecutive days per sample.
        :param max_or_min: 'max' for daily max temperatures, 'min' for daily min.
        :param data_root: Root data directory (default "data").
        :param use_optional_vars: If True (default), include ssrd/tcc when present (7 channels); else 5.
        """
        data = []
        time_index = []

        for year in years:
            for month in range(1, 13):
                era5_data, era5_time_index = load_era5(month, year, data_root=data_root, use_optional_vars=use_optional_vars)
                data.append(era5_data.values)
                time_index.append(era5_time_index)

        data = np.concatenate(data, axis=0)
        self.data, self.mean, self.std = z_score_normalize(data)
        self.n_channels = self.data.shape[2]

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

        seq = self.data[start_idx: start_idx + self.window_size] # [Days, Steps, Channels, Lat, Long]

        x = torch.from_numpy(seq)

        x = x.permute(2, 0, 1, 3, 4)

        C, D, S, H, W = x.shape
        x = x.reshape(C, D * S, H, W) # [Channels, Time (Days * Steps), Lat, Long]

        previous_day_target_norm = x[0, -nhours:, :, :].max().item()
        
        previous_day_target = previous_day_target_norm * self.std[0] + self.mean[0] - 273.15 # Unnormalizes, converts K to C

        residual = target_val - previous_day_target

        target = torch.tensor(residual, dtype=torch.float32)
        baseline = torch.tensor(previous_day_target, dtype=torch.float32)
        return x, target, baseline


if __name__ == "__main__":
    #data = central_park_daily_max_min()
    #print(data["DATE"][0])
    data, _ = load_era5(6, 2015)
    print(data)