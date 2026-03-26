"""
GFS Research Pipeline for Kalshi NYC Temperature Markets
=========================================================
Builds a research dataset with engineered features for signal discovery.

Dataset: One row per (forecast_init_time, market_date)
Target: Predict GFS forecast errors for Central Park daily high/low

Features:
- Forecasted daily high/low (from GFS)
- Atmospheric variables at Central Park (wind, pressure, upper-air)
- Engineered features (wind speed, direction, temperature spread, etc.)
- Temporal features (month, day of year, lead time)
"""

import requests
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

# Central Park coordinates
CP_LAT = 40.78333
CP_LON = -73.96667
CP_LON_360 = 360 + CP_LON  # GFS uses 0-360 longitude

# Timezone
NYC_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# NOMADS bounding box (small region around NYC)
BBOX = {'left_lon': 284, 'right_lon': 289, 'top_lat': 42, 'bottom_lat': 39}

# GFS variables to download
VARIABLES = ['TMP', 'UGRD', 'VGRD', 'PRMSL', 'HGT']
LEVELS = ['2_m_above_ground', '10_m_above_ground', 'mean_sea_level', '850_mb', '500_mb']

# Forecast hours for day-ahead (00Z run → next day's high/low)
FORECAST_HOURS = [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def build_gfs_url(init_date: datetime, cycle: str, forecast_hour: int) -> str:
    """Build NOMADS filter URL."""
    date_str = init_date.strftime('%Y%m%d')
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    
    params = [
        f"file=gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}",
        f"dir=/gfs.{date_str}/{cycle}/atmos",
        "subregion=",
        f"leftlon={BBOX['left_lon']}",
        f"rightlon={BBOX['right_lon']}",
        f"toplat={BBOX['top_lat']}",
        f"bottomlat={BBOX['bottom_lat']}",
    ]
    for var in VARIABLES:
        params.append(f"var_{var}=on")
    for lev in LEVELS:
        params.append(f"lev_{lev}=on")
    
    return base_url + "?" + "&".join(params)


def download_gfs_hour(init_date: datetime, forecast_hour: int,
                      output_dir: str = "data/gfs", cycle: str = '00') -> str:
    """Download a single GFS forecast hour."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"gfs_{init_date.strftime('%Y%m%d')}_{cycle}z_f{forecast_hour:03d}.grib2"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    url = build_gfs_url(init_date, cycle, forecast_hour)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    return filepath


# =============================================================================
# DATA LOADING - ALL VARIABLES
# =============================================================================

def load_all_variables(filepath: str) -> dict:
    """
    Load all variables from a GRIB2 file.
    Returns dict of variable_name → value at Central Park (in standard units).
    """
    data = {}
    
    # 2m temperature (K → C)
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        var = ds['t2m'] if 't2m' in ds else ds['t']
        val = var.sel(latitude=CP_LAT, longitude=CP_LON_360, method='nearest').values
        data['t2m_celsius'] = float(val) - 273.15
    except:
        pass
    
    # 10m winds (m/s)
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
        u = ds['u10'].sel(latitude=CP_LAT, longitude=CP_LON_360, method='nearest').values
        v = ds['v10'].sel(latitude=CP_LAT, longitude=CP_LON_360, method='nearest').values
        data['u10_mps'] = float(u)
        data['v10_mps'] = float(v)
    except:
        pass
    
    # Mean sea level pressure (Pa → hPa)
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}})
        var = ds['prmsl'] if 'prmsl' in ds else ds['msl']
        val = var.sel(latitude=CP_LAT, longitude=CP_LON_360, method='nearest').values
        data['mslp_hpa'] = float(val) / 100.0
    except:
        pass
    
    # Pressure level data
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
        
        # 850mb temperature (K → C)
        if 't' in ds:
            val = ds['t'].sel(latitude=CP_LAT, longitude=CP_LON_360, 
                              isobaricInhPa=850, method='nearest').values
            data['t850_celsius'] = float(val) - 273.15
        
        # 500mb geopotential height (m)
        if 'gh' in ds:
            val = ds['gh'].sel(latitude=CP_LAT, longitude=CP_LON_360,
                               isobaricInhPa=500, method='nearest').values
            data['z500_meters'] = float(val)
    except:
        pass
    
    return data


# =============================================================================
# TIME UTILITIES
# =============================================================================

def get_valid_time(init_date: datetime, forecast_hour: int, cycle: str = '00') -> datetime:
    """Compute valid time (UTC) for a forecast."""
    init_time = datetime(init_date.year, init_date.month, init_date.day,
                         int(cycle), 0, 0, tzinfo=UTC_TZ)
    return init_time + timedelta(hours=forecast_hour)


def valid_time_to_nyc_date(valid_time: datetime):
    """Convert UTC valid time to NYC local date."""
    return valid_time.astimezone(NYC_TZ).date()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_wind_features(u10: float, v10: float) -> dict:
    """
    Compute wind-derived features.
    
    u10: eastward wind component (positive = wind blowing east, i.e., FROM west)
    v10: northward wind component (positive = wind blowing north, i.e., FROM south)
    
    Returns:
        wind_speed_mps: magnitude of wind vector
        wind_direction_deg: meteorological convention (direction wind is FROM, 0=N, 90=E)
        is_onshore: whether wind is coming from the ocean (E/SE for NYC)
    """
    # Wind speed
    wind_speed = math.sqrt(u10**2 + v10**2)
    
    # Wind direction (meteorological convention: direction wind is coming FROM)
    # atan2 gives the direction wind is blowing TO, so we add 180°
    # Then convert to meteorological: 0° = North, 90° = East
    if wind_speed < 0.1:
        wind_dir = np.nan  # Calm winds, direction undefined
    else:
        # Direction wind is blowing TO (math convention)
        dir_to = math.degrees(math.atan2(u10, v10))  # Note: atan2(x,y) not atan2(y,x)
        # Convert to direction FROM
        dir_from = (dir_to + 180) % 360
        wind_dir = dir_from
    
    # Onshore indicator for NYC
    # Ocean is to the east/southeast. Onshore = wind coming from ~45° to ~180° (NE to S)
    if pd.isna(wind_dir):
        is_onshore = np.nan
    else:
        is_onshore = 1 if (45 <= wind_dir <= 180) else 0
    
    return {
        'wind_speed_mps': wind_speed,
        'wind_direction_deg': wind_dir,
        'is_onshore_wind': is_onshore,
    }


def compute_temperature_spread(t2m: float, t850: float) -> float:
    """
    Compute temperature spread: T2m - T850.
    
    Positive = surface warmer than 850mb (normal daytime, stable conditions)
    Negative = surface colder than 850mb (inversion, unstable)
    
    This indicates atmospheric stability and can affect forecast skill.
    """
    if pd.isna(t2m) or pd.isna(t850):
        return np.nan
    return t2m - t850


def compute_temporal_features(market_date) -> dict:
    """
    Compute temporal features for a market date.
    """
    if isinstance(market_date, datetime):
        d = market_date
    else:
        d = datetime.combine(market_date, datetime.min.time())
    
    return {
        'month': d.month,
        'day_of_year': d.timetuple().tm_yday,
        'is_winter': 1 if d.month in [12, 1, 2] else 0,
        'is_summer': 1 if d.month in [6, 7, 8] else 0,
    }


def categorize_wind_regime(wind_speed: float) -> str:
    """Categorize wind speed into regimes for analysis."""
    if pd.isna(wind_speed):
        return 'unknown'
    elif wind_speed < 2:
        return 'calm'
    elif wind_speed < 5:
        return 'light'
    elif wind_speed < 10:
        return 'moderate'
    else:
        return 'strong'


# =============================================================================
# BUILD FORECAST TIMESERIES
# =============================================================================

def build_forecast_timeseries(init_date: datetime, cycle: str = '00',
                               output_dir: str = 'data/gfs') -> pd.DataFrame:
    """
    Download all forecast hours and build timeseries of all variables.
    
    Returns DataFrame with one row per forecast hour, including:
    - forecast_hour, valid_time_utc, nyc_date
    - All atmospheric variables
    - Engineered features
    """
    records = []
    
    for fh in FORECAST_HOURS:
        try:
            filepath = download_gfs_hour(init_date, fh, output_dir, cycle)
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
            
            # Add wind features
            if 'u10_mps' in data and 'v10_mps' in data:
                wind_features = compute_wind_features(data['u10_mps'], data['v10_mps'])
                record.update(wind_features)
            
            # Add temperature spread
            if 't2m_celsius' in data and 't850_celsius' in data:
                record['temp_spread_surface_850'] = compute_temperature_spread(
                    data['t2m_celsius'], data['t850_celsius']
                )
            
            records.append(record)
            
        except Exception as e:
            pass  # Skip failed hours
    
    return pd.DataFrame(records)


# =============================================================================
# AGGREGATE TO DAILY
# =============================================================================

def aggregate_daily(ts_df: pd.DataFrame, target_date) -> dict:
    """
    Aggregate forecast timeseries to daily values for a target NYC date.
    
    Returns dict with:
    - Forecasted high/low (max/min of t2m)
    - Mean values of other features over the day
    """
    day_data = ts_df[ts_df['nyc_date'] == target_date]
    
    if len(day_data) == 0:
        return None
    
    result = {
        'n_forecast_hours': len(day_data),
        
        # Daily high/low from 2m temperature
        'forecasted_high_celsius': day_data['t2m_celsius'].max(),
        'forecasted_low_celsius': day_data['t2m_celsius'].min(),
        
        # Mean lead time (hours ahead)
        'mean_lead_time_hours': day_data['forecast_hour'].mean(),
    }
    
    # Mean of atmospheric features over the day
    for col in ['u10_mps', 'v10_mps', 'mslp_hpa', 't850_celsius', 'z500_meters',
                'wind_speed_mps', 'wind_direction_deg', 'temp_spread_surface_850']:
        if col in day_data.columns:
            result[f'mean_{col}'] = day_data[col].mean()
    
    # Onshore wind: fraction of hours with onshore wind
    if 'is_onshore_wind' in day_data.columns:
        result['onshore_wind_fraction'] = day_data['is_onshore_wind'].mean()
    
    return result


# =============================================================================
# ACTUALS LOADING
# =============================================================================

def load_actuals(filepath: str) -> pd.DataFrame:
    """
    Load Central Park actual high/low temperatures.
    Expects CSV with: DATE, TMAX, TMIN (temperatures in Fahrenheit)
    """
    df = pd.read_csv(filepath, parse_dates=['DATE'])
    df = df.rename(columns={'DATE': 'date'})
    
    # Convert F to C
    df['actual_high_celsius'] = (df['TMAX'] - 32) * 5/9
    df['actual_low_celsius'] = (df['TMIN'] - 32) * 5/9
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df[['date', 'actual_high_celsius', 'actual_low_celsius']]


# =============================================================================
# MAIN PIPELINE: BUILD RESEARCH DATASET
# =============================================================================

def process_single_init(init_date: datetime, actuals_df: pd.DataFrame = None,
                        cycle: str = '00', output_dir: str = 'data/gfs',
                        verbose: bool = True) -> dict:
    """
    Process one GFS initialization → one dataset row with all features.
    """
    if verbose:
        print(f"Processing {init_date.strftime('%Y-%m-%d')} {cycle}Z...", end=" ")
    
    # Build forecast timeseries
    ts_df = build_forecast_timeseries(init_date, cycle, output_dir)
    
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


def build_research_dataset(init_dates: list, actuals_df: pd.DataFrame = None,
                           cycle: str = '00', output_dir: str = 'data/gfs') -> pd.DataFrame:
    """
    Build the full research dataset for multiple GFS initializations.
    """
    rows = []
    
    for init_date in init_dates:
        try:
            row = process_single_init(init_date, actuals_df, cycle, output_dir)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"Failed {init_date}: {e}")
    
    return pd.DataFrame(rows)


# =============================================================================
# DATASET SCHEMA
# =============================================================================

DATASET_SCHEMA = """
RESEARCH DATASET SCHEMA
=======================

One row per (init_date, market_date) pair.

IDENTIFIERS:
  init_date                    - GFS initialization date
  init_cycle                   - Forecast cycle ('00', '06', etc.)
  market_date                  - NYC local date being forecasted

TARGETS:
  forecasted_high_celsius      - GFS max 2m temp over NYC day
  forecasted_low_celsius       - GFS min 2m temp over NYC day
  actual_high_celsius          - Actual Central Park daily high
  actual_low_celsius           - Actual Central Park daily low
  error_high_celsius           - actual - forecast (+ = GFS cold bias)
  error_low_celsius            - actual - forecast (+ = GFS cold bias)

RAW ATMOSPHERIC FEATURES (mean over forecast hours):
  mean_u10_mps                 - Eastward wind component
  mean_v10_mps                 - Northward wind component
  mean_mslp_hpa                - Mean sea level pressure
  mean_t850_celsius            - 850mb temperature
  mean_z500_meters             - 500mb geopotential height

ENGINEERED FEATURES:
  mean_wind_speed_mps          - Wind speed = sqrt(u^2 + v^2)
  mean_wind_direction_deg      - Meteorological wind direction (FROM)
  mean_temp_spread_surface_850 - T2m - T850 (stability indicator)
  onshore_wind_fraction        - Fraction of hours with onshore wind
  wind_regime                  - Categorical: calm/light/moderate/strong

TEMPORAL FEATURES:
  month                        - Month (1-12)
  day_of_year                  - Day of year (1-366)
  is_winter                    - 1 if Dec/Jan/Feb
  is_summer                    - 1 if Jun/Jul/Aug
  mean_lead_time_hours         - Mean forecast lead time

METADATA:
  n_forecast_hours             - Number of 3-hourly forecasts used
"""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GFS RESEARCH PIPELINE FOR KALSHI NYC TEMPERATURE MARKETS")
    print("=" * 70)
    
    # Create mock actuals for demonstration
    mock_actuals = pd.DataFrame({
        'date': [(datetime.now() - timedelta(days=i)).date() for i in range(1, 12)],
        'actual_high_celsius': [14.0, 16.0, 12.0, 18.0, 20.0, 15.0, 11.0, 17.0, 19.0, 13.0, 10.0],
        'actual_low_celsius': [6.0, 8.0, 4.0, 10.0, 12.0, 7.0, 3.0, 9.0, 11.0, 5.0, 2.0],
    })
    
    print("\nMock actuals (for demo):")
    print(mock_actuals.head())
    
    # Process last several days (NOMADS has ~10 days)
    init_dates = [datetime.now() - timedelta(days=i) for i in range(3, 10)]
    
    print(f"\nProcessing {len(init_dates)} GFS initializations...")
    print("-" * 70)
    
    df = build_research_dataset(init_dates, mock_actuals)
    
    # Show dataset
    print("\n" + "=" * 70)
    print("RESEARCH DATASET (first few rows)")
    print("=" * 70)
    
    # Select key columns for display
    display_cols = [
        'init_date', 'market_date', 
        'forecasted_high_celsius', 'actual_high_celsius', 'error_high_celsius',
        'mean_wind_speed_mps', 'wind_regime', 'onshore_wind_fraction',
        'mean_temp_spread_surface_850', 'month'
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df[display_cols].to_string())
    
    # Show full schema
    print("\n" + "=" * 70)
    print("ALL COLUMNS IN DATASET")
    print("=" * 70)
    for col in df.columns:
        print(f"  {col}")
    
    print("\n" + DATASET_SCHEMA)
