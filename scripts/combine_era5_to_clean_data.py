#!/usr/bin/env python3
"""
Combine ERA5 .nc and .grib files into one netCDF per year for easy loading.
Output: data/clean_data/era5_YYYY.nc (all variables, all hours, 64x64 lat/lon).

Requires: netCDF4 or h5netcdf (pip install netCDF4). Optional: cfgrib to merge .grib vars.
Run from project root: python scripts/combine_era5_to_clean_data.py
Processing all years can take a long time; use --years 2015 2022 to test.
"""
import os
import re
import argparse
from pathlib import Path

import xarray as xr
import numpy as np


# Default paths (relative to project root)
DATA_ROOT = "data"
RAW_SUBFOLDER = "era5_raw"
CLEAN_SUBFOLDER = "clean_data"
LAT_SLICE = slice(0, 64)
LON_SLICE = slice(0, 64)

# Time dimension name in ERA5 files
TIME_DIM = "valid_time"


def find_data_root():
    """Resolve data root: script dir is scripts/, project root is parent."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / DATA_ROOT


def discover_files(data_root: Path):
    """Find all .nc and .grib files under data_root, grouped by year."""
    raw_dir = data_root / RAW_SUBFOLDER
    if not raw_dir.exists():
        return {}, {}

    nc_by_year = {}   # year -> list of Path
    grib_by_year = {} # year -> list of Path

    # era5_YYYY_MM.nc or era5_YYYY_MM.grib
    nc_month_pat = re.compile(r"era5_(\d{4})_(\d{2})\.nc$", re.IGNORECASE)
    # era5_*_YYYY.grib
    grib_year_pat = re.compile(r"era5_.*_?(\d{4})\.grib", re.IGNORECASE)

    for root, _, files in os.walk(raw_dir):
        root_path = Path(root)
        for f in files:
            path = root_path / f
            if f.lower().endswith(".nc"):
                m = nc_month_pat.match(f)
                if m:
                    year = int(m.group(1))
                    nc_by_year.setdefault(year, []).append(path)
            elif f.lower().endswith(".grib"):
                m = grib_year_pat.match(f)
                if m:
                    year = int(m.group(1))
                    grib_by_year.setdefault(year, []).append(path)

    # Sort nc files by month within each year
    for year in nc_by_year:
        nc_by_year[year].sort(key=lambda p: (p.name,))

    return nc_by_year, grib_by_year


def _get_engine():
    """Prefer netcdf4 for .nc; required for ERA5 netCDF from CDS."""
    try:
        import netCDF4  # noqa: F401
        return "netcdf4"
    except ImportError:
        pass
    try:
        import h5netcdf  # noqa: F401
        return "h5netcdf"
    except ImportError:
        pass
    return None


def combine_nc_for_year(file_paths: list, trim: bool = True) -> xr.Dataset:
    """Open and concatenate netCDF files along time; optionally trim to 64x64."""
    if not file_paths:
        return xr.Dataset()

    paths_str = [str(p) for p in file_paths]
    engine = _get_engine()
    if engine is None:
        print("  Warning: install netCDF4 or h5netcdf to read .nc files (e.g. pip install netCDF4)")
        return xr.Dataset()

    # Detect time dimension from first file (CDS uses "time", others "valid_time")
    try:
        with xr.open_dataset(paths_str[0], engine=engine) as first:
            time_dim = TIME_DIM if TIME_DIM in first.dims else "time"
    except Exception:
        time_dim = TIME_DIM

    try:
        ds = xr.open_mfdataset(
            paths_str,
            concat_dim=time_dim,
            combine="nested",
            compat="no_conflicts",
            engine=engine,
        )
    except Exception as e:
        print(f"  Warning: failed to open netCDF: {e}")
        return xr.Dataset()

    if trim and "latitude" in ds.dims and "longitude" in ds.dims:
        lat_len = ds.sizes.get("latitude", 0)
        lon_len = ds.sizes.get("longitude", 0)
        if lat_len >= 64 and lon_len >= 64:
            ds = ds.isel(latitude=LAT_SLICE, longitude=LON_SLICE)
    ds = ds.sortby(time_dim)
    return ds


def try_merge_grib(ds_nc: xr.Dataset, grib_paths: list) -> xr.Dataset:
    """If cfgrib is available, open GRIB and merge non-overlapping variables."""
    try:
        import cfgrib
    except ImportError:
        return ds_nc

    for gpath in grib_paths:
        try:
            ds_g = xr.open_dataset(gpath, engine="cfgrib")
            # Merge only if we have compatible coords (same lat/lon/time names)
            common = set(ds_nc.coords) & set(ds_g.coords)
            if common:
                # Drop from grib any var that exists in nc; then merge
                to_merge = ds_g.drop_vars([v for v in ds_g.data_vars if v in ds_nc.data_vars], errors="ignore")
                if to_merge.data_vars:
                    ds_nc = xr.merge([ds_nc, to_merge], compat="no_conflicts", join="outer")
            ds_g.close()
        except Exception as e:
            print(f"  Warning: could not merge GRIB {gpath}: {e}")
    return ds_nc


def main():
    parser = argparse.ArgumentParser(description="Combine ERA5 .nc/.grib into one file per year.")
    parser.add_argument("--data-root", default=None, help=f"Override data root (default: project_root/{DATA_ROOT})")
    parser.add_argument("--no-trim", action="store_true", help="Do not trim lat/lon to 64x64")
    parser.add_argument("--nc-only", action="store_true", help="Only combine .nc files; skip merging .grib")
    parser.add_argument("--years", nargs="*", type=int, help="Process only these years (default: all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing clean_data files")
    args = parser.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = find_data_root()

    out_dir = data_root / CLEAN_SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_root}")
    print(f"Output:    {out_dir}")

    nc_by_year, grib_by_year = discover_files(data_root)
    all_years = sorted(set(nc_by_year) | set(grib_by_year))
    if not all_years:
        print("No .nc or .grib files found under data/era5_raw.")
        return

    if args.years:
        all_years = [y for y in all_years if y in args.years]
    print(f"Years to process: {all_years}")

    for year in all_years:
        nc_paths = nc_by_year.get(year, [])
        grib_paths = grib_by_year.get(year, [])

        if not nc_paths and not grib_paths:
            continue

        out_path = out_dir / f"era5_{year}.nc"
        if out_path.exists() and not args.force:
            print(f"\n{year}: {out_path.name} already exists, skipping (use --force to overwrite)")
            continue

        print(f"\n{year}: {len(nc_paths)} .nc, {len(grib_paths)} .grib -> {out_path.name}")

        if nc_paths:
            ds = combine_nc_for_year(nc_paths, trim=not args.no_trim)
            if ds.dims and len(ds.data_vars) > 0:
                if grib_paths and not args.nc_only:
                    ds = try_merge_grib(ds, grib_paths)
                # Drop auxiliary vars that cause serialization warnings (e.g. expver)
                for v in list(ds.variables):
                    if v not in ds.dims and ds.variables[v].dtype == object:
                        ds = ds.drop_vars(v, errors="ignore")
                ds.to_netcdf(out_path)
                print(f"  Wrote {out_path} ({list(ds.data_vars.keys())})")
                ds.close()
            else:
                print(f"  Skipped (no data).")
        elif grib_paths and not args.nc_only:
            try:
                import cfgrib
                ds = xr.open_mfdataset(
                    [str(p) for p in grib_paths],
                    engine="cfgrib",
                    combine="by_coords",
                )
                if "latitude" in ds.dims and "longitude" in ds.dims and not args.no_trim:
                    ds = ds.isel(latitude=LAT_SLICE, longitude=LON_SLICE)
                ds.to_netcdf(out_path)
                print(f"  Wrote {out_path} (from GRIB: {list(ds.data_vars.keys())})")
                ds.close()
            except Exception as e:
                print(f"  Failed to combine GRIB for {year}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
