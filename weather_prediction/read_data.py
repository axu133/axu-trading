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

    time_index = batched['valid_time'].values

    return data, time_index

if __name__ == "__main__":
    #data = central_park_daily_max_min()
    #print(data["DATE"][0])
    data, _ = load_era5(6, 2015)
    print(data)