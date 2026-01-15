import pandas as pd
import os
import xarray as xr

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
    Load ERA5 data for a specific month and year. Returns 6-hourly data as a pandas DataFrame.
    """
    file_path = os.path.join("data", "era5_raw", str(year), f"era5_{year}_{month:02d}.nc")
    ds = xr.open_dataset(file_path)

    subset = ds.sel(valid_time = ds.valid_time.dt.hour.isin([0, 6, 12, 18]))

    return subset

if __name__ == "__main__":
    #data = central_park_daily_max_min()
    #print(data["DATE"][0])
    data = load_era5(1, 2015)[['t2m', 'u10', 'v10', 'msl', 'd2m']].to_array(dim='channel')
    data = data.transpose('valid_time', 'channel', 'latitude', 'longitude')
    print(data)