"""
Historical GFS Data Fetcher (AWS Open Data)
============================================
Downloads historical GFS forecast data from AWS Open Data Registry.

AWS Bucket: s3://noaa-gfs-bdp-pds/
- Free, no authentication required
- Contains GFS data from ~2021 onwards
- Same GRIB2 format as NOMADS

Usage:
    from gfs_historical import download_historical_gfs, build_historical_dataset
    
    # Download a single date
    files = download_historical_gfs(datetime(2024, 6, 15))
    
    # Build dataset for a date range
    df = build_historical_dataset(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        actuals_df=actuals_df
    )
"""

import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import from research pipeline
from gfs_research_pipeline import (
    load_all_variables,
    get_valid_time,
    valid_time_to_nyc_date,
    compute_wind_features,
    compute_temperature_spread,
    compute_temporal_features,
    categorize_wind_regime,
    aggregate_daily,
    build_forecast_timeseries,
    CP_LAT, CP_LON_360, FORECAST_HOURS
)

# AWS Open Data bucket (public, no auth needed)
AWS_GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def build_aws_gfs_url(init_date: datetime, cycle: str, forecast_hour: int) -> str:
    """
    Build URL to download GFS file from AWS Open Data.
    
    AWS structure: gfs.{YYYYMMDD}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{fhour}
    
    Note: AWS has the FULL global files (not filtered like NOMADS).
    Files are ~30-50 MB each vs ~10 KB from NOMADS filter.
    """
    date_str = init_date.strftime('%Y%m%d')
    filename = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
    
    url = f"{AWS_GFS_BASE}/gfs.{date_str}/{cycle}/atmos/{filename}"
    return url


def download_aws_gfs_hour(init_date: datetime, forecast_hour: int,
                          output_dir: str = "data/gfs_historical", 
                          cycle: str = '00') -> str:
    """
    Download a single GFS forecast hour from AWS.
    
    WARNING: AWS files are FULL global files (~30-50 MB each).
    For research, consider downloading once and caching.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = init_date.strftime('%Y%m%d')
    filename = f"gfs_{date_str}_{cycle}z_f{forecast_hour:03d}.grib2"
    filepath = os.path.join(output_dir, filename)
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        return filepath
    
    url = build_aws_gfs_url(init_date, cycle, forecast_hour)
    
    print(f"  Downloading f{forecast_hour:03d} from AWS (~30-50 MB)...")
    
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    
    # Stream to file (large files)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Saved: {filepath} ({file_size_mb:.1f} MB)")
    
    return filepath


def check_aws_availability(init_date: datetime, cycle: str = '00') -> bool:
    """Check if GFS data is available on AWS for a given date."""
    url = build_aws_gfs_url(init_date, cycle, 24)
    
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def download_historical_gfs(init_date: datetime, 
                            output_dir: str = "data/gfs_historical",
                            cycle: str = '00',
                            forecast_hours: list = None) -> list:
    """
    Download all forecast hours for a single GFS initialization from AWS.
    
    Args:
        init_date: GFS initialization date
        output_dir: Where to save files
        cycle: Forecast cycle ('00', '06', '12', '18')
        forecast_hours: List of forecast hours (default: FORECAST_HOURS)
    
    Returns:
        List of (forecast_hour, filepath) tuples
    """
    if forecast_hours is None:
        forecast_hours = FORECAST_HOURS
    
    print(f"Downloading GFS {init_date.strftime('%Y-%m-%d')} {cycle}Z from AWS...")
    
    filepaths = []
    for fh in forecast_hours:
        try:
            fp = download_aws_gfs_hour(init_date, fh, output_dir, cycle)
            filepaths.append((fh, fp))
        except Exception as e:
            print(f"  Failed f{fh:03d}: {e}")
    
    return filepaths


def build_historical_timeseries(init_date: datetime, cycle: str = '00',
                                 output_dir: str = 'data/gfs_historical') -> pd.DataFrame:
    """
    Download historical GFS and build forecast timeseries (same as research pipeline).
    Uses AWS instead of NOMADS.
    """
    records = []
    
    for fh in FORECAST_HOURS:
        try:
            filepath = download_aws_gfs_hour(init_date, fh, output_dir, cycle)
            data = load_all_variables(filepath)
            
            if 't2m_celsius' not in data:
                continue
            
            valid_time = get_valid_time(init_date, fh, cycle)
            nyc_date = valid_time_to_nyc_date(valid_time)
            
            record = {
                'forecast_hour': fh,
                'valid_time_utc': valid_time,
                'nyc_date': nyc_date,
                **data,
            }
            
            if 'u10_mps' in data and 'v10_mps' in data:
                wind_features = compute_wind_features(data['u10_mps'], data['v10_mps'])
                record.update(wind_features)
            
            if 't2m_celsius' in data and 't850_celsius' in data:
                record['temp_spread_surface_850'] = compute_temperature_spread(
                    data['t2m_celsius'], data['t850_celsius']
                )
            
            records.append(record)
            
        except Exception as e:
            print(f"  Warning: f{fh:03d} failed: {e}")
    
    return pd.DataFrame(records)


def process_historical_init(init_date: datetime, actuals_df: pd.DataFrame = None,
                            cycle: str = '00', output_dir: str = 'data/gfs_historical',
                            verbose: bool = True) -> dict:
    """
    Process one historical GFS initialization → one dataset row.
    Same logic as research pipeline but uses AWS data.
    """
    if verbose:
        print(f"Processing {init_date.strftime('%Y-%m-%d')} {cycle}Z...", end=" ")
    
    # Build forecast timeseries from AWS
    ts_df = build_historical_timeseries(init_date, cycle, output_dir)
    
    if len(ts_df) == 0:
        if verbose:
            print("No data")
        return None
    
    # Target market date (tomorrow for 00Z)
    target_date = (init_date + timedelta(days=1)).date()
    
    # Aggregate to daily
    daily = aggregate_daily(ts_df, target_date)
    
    if daily is None:
        if verbose:
            print("No data for target date")
        return None
    
    # Build row with metadata
    row = {
        'init_date': init_date.date() if isinstance(init_date, datetime) else init_date,
        'init_cycle': cycle,
        'market_date': target_date,
        **daily,
    }
    
    # Add temporal features
    temporal = compute_temporal_features(target_date)
    row.update(temporal)
    
    # Add wind regime category
    if 'mean_wind_speed_mps' in row:
        row['wind_regime'] = categorize_wind_regime(row['mean_wind_speed_mps'])
    
    # Merge with actuals
    if actuals_df is not None:
        actual_row = actuals_df[actuals_df['date'] == target_date]
        if len(actual_row) > 0:
            row['actual_high_celsius'] = actual_row['actual_high_celsius'].values[0]
            row['actual_low_celsius'] = actual_row['actual_low_celsius'].values[0]
            row['error_high_celsius'] = row['actual_high_celsius'] - row['forecasted_high_celsius']
            row['error_low_celsius'] = row['actual_low_celsius'] - row['forecasted_low_celsius']
        else:
            row['actual_high_celsius'] = np.nan
            row['actual_low_celsius'] = np.nan
            row['error_high_celsius'] = np.nan
            row['error_low_celsius'] = np.nan
    
    if verbose:
        print(f"High: {row['forecasted_high_celsius']:.1f}°C, Low: {row['forecasted_low_celsius']:.1f}°C")
    
    return row


def build_historical_dataset(start_date: datetime, end_date: datetime,
                             actuals_df: pd.DataFrame = None,
                             cycle: str = '00',
                             output_dir: str = 'data/gfs_historical') -> pd.DataFrame:
    """
    Build research dataset for a historical date range using AWS GFS data.
    
    Args:
        start_date: First GFS init date to process
        end_date: Last GFS init date to process
        actuals_df: DataFrame with actual temperatures
        cycle: Forecast cycle
        output_dir: Where to save/cache GRIB files
    
    Returns:
        DataFrame with same schema as research pipeline
    
    Example:
        df = build_historical_dataset(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            actuals_df=actuals_df
        )
    """
    rows = []
    
    current = start_date
    total_days = (end_date - start_date).days + 1
    processed = 0
    
    print(f"Building historical dataset: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total days to process: {total_days}")
    print("=" * 60)
    
    while current <= end_date:
        try:
            row = process_historical_init(current, actuals_df, cycle, output_dir, verbose=True)
            if row is not None:
                rows.append(row)
                processed += 1
        except Exception as e:
            print(f"Failed {current.strftime('%Y-%m-%d')}: {e}")
        
        current += timedelta(days=1)
    
    print("=" * 60)
    print(f"Successfully processed: {processed}/{total_days} days")
    
    return pd.DataFrame(rows)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AWS GFS HISTORICAL DATA FETCHER")
    print("=" * 70)
    print()
    print("AWS Bucket: s3://noaa-gfs-bdp-pds/")
    print("Data available: ~2021 onwards")
    print()
    print("WARNING: AWS files are FULL global files (~30-50 MB each).")
    print("         For a month of data, expect ~10-20 GB of downloads.")
    print()
    
    # Check availability for a sample date
    test_date = datetime(2024, 6, 15)
    print(f"Checking availability for {test_date.strftime('%Y-%m-%d')}...")
    
    available = check_aws_availability(test_date)
    print(f"  Available: {available}")
    
    if available:
        print()
        print("Example usage:")
        print("-" * 50)
        print("""
from gfs_historical import build_historical_dataset
from noaa_actuals import fetch_central_park_actuals
from datetime import datetime

# 1. Fetch actuals for your date range
actuals = fetch_central_park_actuals("2024-06-01", "2024-06-30")

# 2. Build historical dataset
df = build_historical_dataset(
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2024, 6, 30),
    actuals_df=actuals
)

# 3. Analyze!
print(df)
""")
