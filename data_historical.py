import os
import cdsapi

c = cdsapi.Client()

DATA_DIR = "data/"
RAW_DIR = "era5_raw/"

OUT_DIR =  os.path.join(DATA_DIR, RAW_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

years = range(1940, 1980)

for year in years:
    year_str = str(year)
    print(f"Downloading data for year {year}")
    year_dir = os.path.join(OUT_DIR, year_str)
    os.makedirs(year_dir, exist_ok=True)

    for month in range(1,13):
        month_str = f"{month:02d}"
        filename = os.path.join(year_dir, f"era5_{year_str}_{month_str}.nc")

        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download.")
            continue

        print(f"Downloading data for {year_str}-{month_str}")
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_dewpoint_temperature',
                        '2m_temperature',
                        'mean_sea_level_pressure',
                    ],
                    'year': year_str,
                    'month': month_str,
                    'day': [
                        '01', '02', '03', '04', '05', '06',
                        '07', '08', '09', '10', '11', '12',
                        '13', '14', '15', '16', '17', '18',
                        '19', '20', '21', '22', '23', '24',
                        '25', '26', '27', '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
                    ],
                    'area': [49, -88, 33, -72],
                },
                filename)
        except Exception as e:
            print(f"Failed to download data for {year_str}-{month_str}: {e}")
            if os.path.exists(filename):
                os.remove(filename)