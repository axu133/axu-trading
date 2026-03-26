"""
NOAA GHCN Daily Data Fetcher
============================
Fetches actual daily high/low temperatures for Central Park from NOAA's
Global Historical Climatology Network (GHCN) Daily dataset.

SETUP:
1. Get a free API token from: https://www.ncdc.noaa.gov/cdo-web/token
2. Set your token as an environment variable:
   export NOAA_API_TOKEN="your_token_here"
   
   Or pass it directly to the functions.

USAGE:
    from noaa_actuals import fetch_central_park_actuals
    
    df = fetch_central_park_actuals(
        start_date="2024-01-01",
        end_date="2024-01-31",
        token="your_token"
    )
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env file on import
load_env_file()

# NOAA API Configuration
NOAA_API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
CENTRAL_PARK_STATION = "GHCND:USW00094728"
DATASET_ID = "GHCND"

# GHCN variables
# TMAX/TMIN are in tenths of degrees Celsius
VARIABLES = ["TMAX", "TMIN"]


def fetch_central_park_actuals(start_date: str, end_date: str, 
                                token: str = None) -> pd.DataFrame:
    """
    Fetch Central Park daily high/low temperatures from NOAA GHCN Daily.
    
    Args:
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
        token: NOAA API token (or set NOAA_API_TOKEN env var)
    
    Returns:
        DataFrame with columns: date, actual_high_celsius, actual_low_celsius
    
    Note:
        - NOAA API limits to 1000 records per request
        - For long date ranges, this function makes multiple requests
        - There may be a 1-2 day lag in data availability
    """
    if token is None:
        token = os.environ.get("NOAA_API_TOKEN")
    
    if not token:
        raise ValueError(
            "NOAA API token required. Get one at: https://www.ncdc.noaa.gov/cdo-web/token\n"
            "Then either:\n"
            "  1. Set NOAA_API_TOKEN environment variable, or\n"
            "  2. Pass token='your_token' to this function"
        )
    
    headers = {"token": token}
    
    # NOAA API has a 1-year max per request, so we may need to chunk
    all_records = []
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Fetch in chunks (NOAA limits to 1 year per request)
    current_start = start
    while current_start <= end:
        current_end = min(current_start + timedelta(days=364), end)
        
        params = {
            "datasetid": DATASET_ID,
            "stationid": CENTRAL_PARK_STATION,
            "datatypeid": ",".join(VARIABLES),
            "startdate": current_start.strftime("%Y-%m-%d"),
            "enddate": current_end.strftime("%Y-%m-%d"),
            "units": "metric",  # Returns Celsius
            "limit": 1000,
        }
        
        print(f"Fetching {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        
        response = requests.get(
            f"{NOAA_API_BASE}/data",
            headers=headers,
            params=params,
            timeout=30
        )
        
        if response.status_code == 429:
            print("Rate limited. Waiting 1 second...")
            time.sleep(1)
            continue
        
        response.raise_for_status()
        data = response.json()
        
        if "results" in data:
            all_records.extend(data["results"])
            print(f"  Retrieved {len(data['results'])} records")
        else:
            print(f"  No data returned")
        
        current_start = current_end + timedelta(days=1)
        time.sleep(0.3)  # Be nice to the API
    
    if not all_records:
        print("No data retrieved. Check your date range and API token.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Pivot to get TMAX and TMIN as columns
    df['date'] = pd.to_datetime(df['date']).dt.date
    df_pivot = df.pivot(index='date', columns='datatype', values='value').reset_index()
    
    # Rename columns
    result = pd.DataFrame({
        'date': df_pivot['date'],
        'actual_high_celsius': df_pivot.get('TMAX', pd.Series([None] * len(df_pivot))),
        'actual_low_celsius': df_pivot.get('TMIN', pd.Series([None] * len(df_pivot))),
    })
    
    # Sort by date
    result = result.sort_values('date').reset_index(drop=True)
    
    return result


def fetch_recent_actuals(days: int = 30, token: str = None) -> pd.DataFrame:
    """
    Fetch the most recent N days of Central Park temperatures.
    
    Note: NOAA data typically has a 1-2 day lag.
    """
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days)
    
    return fetch_central_park_actuals(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        token
    )


def get_station_info(token: str = None) -> dict:
    """
    Get metadata about the Central Park weather station.
    """
    if token is None:
        token = os.environ.get("NOAA_API_TOKEN")
    
    if not token:
        raise ValueError("NOAA API token required")
    
    headers = {"token": token}
    
    response = requests.get(
        f"{NOAA_API_BASE}/stations/{CENTRAL_PARK_STATION}",
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    
    return response.json()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NOAA GHCN Daily Data Fetcher")
    print("=" * 60)
    print()
    print("Central Park Station: USW00094728")
    print()
    
    # Check for token
    token = os.environ.get("NOAA_API_TOKEN")
    
    if not token:
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("""
To use this script, you need a free NOAA API token:

1. Go to: https://www.ncdc.noaa.gov/cdo-web/token
2. Enter your email and request a token
3. You'll receive the token via email (usually within minutes)

Then either:

Option A: Set environment variable (recommended)
    export NOAA_API_TOKEN="your_token_here"
    python noaa_actuals.py

Option B: Use in code
    from noaa_actuals import fetch_central_park_actuals
    df = fetch_central_park_actuals("2024-01-01", "2024-01-31", token="your_token")
""")
    else:
        print("Token found. Fetching recent data...\n")
        
        try:
            # Fetch last 14 days
            df = fetch_recent_actuals(days=14, token=token)
            
            print("\n" + "=" * 60)
            print("CENTRAL PARK DAILY TEMPERATURES (Last 14 days)")
            print("=" * 60)
            print(df.to_string(index=False))
            
            print("\n" + "=" * 60)
            print("USAGE IN YOUR PIPELINE")
            print("=" * 60)
            print("""
# In your research pipeline:
from noaa_actuals import fetch_central_park_actuals

# Fetch actuals for your date range
actuals_df = fetch_central_park_actuals("2024-01-01", "2024-03-25")

# Use with research pipeline
from gfs_research_pipeline import build_research_dataset
df = build_research_dataset(init_dates, actuals_df)
""")
            
        except Exception as e:
            print(f"Error: {e}")
