#!/usr/bin/env python3
"""
Combine ERA5 raw data (12 monthly .nc + 1 radiation/cloud .grib per year) into one netCDF per year
for easy loading. Output: data/clean_data/era5_YYYY.nc (all variables, 64x64 lat/lon).

- Skips a year if era5_YYYY.nc already exists (safe to stop/restart).
- Writes to a temp file then atomically renames, so partial writes never appear as complete.
- Default years exclude before 1980 (use --min-year to change).
Requires: netCDF4 or h5netcdf. Optional: cfgrib to merge .grib (ssrd, tcc).
Dask: xarray may warn "dask not available"; without dask it loads files into memory (fine for this script).
"""
import os
import re
import argparse
from pathlib import Path

import xarray as xr
import numpy as np


DATA_ROOT = "data"
RAW_SUBFOLDER = "era5_raw"
CLEAN_SUBFOLDER = "clean_data"
LAT_SLICE = slice(0, 64)
LON_SLICE = slice(0, 64)
TIME_DIM = "valid_time"
TEMP_SUFFIX = ".tmp"
MIN_YEAR_DEFAULT = 1980  # exclude 1940 etc. unless --years or --min-year set


def find_data_root():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / DATA_ROOT


def _get_engine():
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


def discover_files(data_root: Path):
    """Find .nc (era5_YYYY_MM.nc) and .grib (era5_ssrd_tcc_YYYY.grib) per year."""
    raw_dir = data_root / RAW_SUBFOLDER
    if not raw_dir.exists():
        return {}, {}

    nc_by_year = {}
    grib_by_year = {}
    nc_pat = re.compile(r"era5_(\d{4})_(\d{2})\.nc$", re.IGNORECASE)
    grib_pat = re.compile(r"era5_.*_?(\d{4})\.grib", re.IGNORECASE)

    for root, _, files in os.walk(raw_dir):
        root_path = Path(root)
        for f in files:
            path = root_path / f
            if f.lower().endswith(".nc"):
                m = nc_pat.match(f)
                if m:
                    year = int(m.group(1))
                    nc_by_year.setdefault(year, []).append(path)
            elif f.lower().endswith(".grib"):
                m = grib_pat.match(f)
                if m:
                    year = int(m.group(1))
                    grib_by_year.setdefault(year, []).append(root_path / f)

    for year in nc_by_year:
        nc_by_year[year].sort(key=lambda p: p.name)
    return nc_by_year, grib_by_year


def combine_nc_for_year(file_paths: list, trim: bool = True, engine: str = None) -> xr.Dataset:
    """Concatenate monthly netCDFs along time; optionally trim to 64x64."""
    if not file_paths:
        return xr.Dataset()
    paths_str = [str(p) for p in file_paths]
    try:
        with xr.open_dataset(paths_str[0], engine=engine) as first:
            time_dim = TIME_DIM if TIME_DIM in first.dims else "time"
    except Exception:
        time_dim = TIME_DIM

    # chunks=None loads into memory (avoids "dask not available" when dask isn't installed)
    ds = xr.open_mfdataset(
        paths_str,
        concat_dim=time_dim,
        combine="nested",
        compat="no_conflicts",
        engine=engine,
        chunks=None,
    )
    if trim and "latitude" in ds.dims and "longitude" in ds.dims:
        lat_len = ds.sizes.get("latitude", 0)
        lon_len = ds.sizes.get("longitude", 0)
        if lat_len >= 64 and lon_len >= 64:
            ds = ds.isel(latitude=LAT_SLICE, longitude=LON_SLICE)
    ds = ds.sortby(time_dim)
    return ds


def try_merge_grib(ds_nc: xr.Dataset, grib_paths: list) -> xr.Dataset:
    """Merge GRIB variables (e.g. ssrd, tcc) into the dataset if compatible."""
    try:
        import cfgrib
    except ImportError:
        return ds_nc
    for gpath in grib_paths:
        try:
            ds_g = xr.open_dataset(str(gpath), engine="cfgrib")
            common = set(ds_nc.coords) & set(ds_g.coords)
            if common:
                to_merge = ds_g.drop_vars(
                    [v for v in ds_g.data_vars if v in ds_nc.data_vars], errors="ignore"
                )
                if to_merge.data_vars:
                    ds_nc = xr.merge([ds_nc, to_merge], compat="no_conflicts", join="outer")
            ds_g.close()
        except Exception as e:
            print(f"  Warning: could not merge GRIB {gpath.name}: {e}")
    return ds_nc


def drop_object_vars(ds: xr.Dataset) -> xr.Dataset:
    """Drop variables with dtype object to avoid netCDF serialization issues."""
    for v in list(ds.variables):
        if v not in ds.dims and getattr(ds.variables[v], "dtype", None) == object:
            ds = ds.drop_vars(v, errors="ignore")
    return ds


def main():
    parser = argparse.ArgumentParser(description="Combine ERA5 raw files into one netCDF per year.")
    parser.add_argument("--data-root", default=None, help=f"Override data root (default: project/{DATA_ROOT})")
    parser.add_argument("--no-trim", action="store_true", help="Do not trim lat/lon to 64x64")
    parser.add_argument("--nc-only", action="store_true", help="Only combine .nc; skip merging .grib")
    parser.add_argument("--years", nargs="*", type=int, help="Process only these years (overrides --min-year)")
    parser.add_argument("--min-year", type=int, default=MIN_YEAR_DEFAULT, help=f"Minimum year when processing all (default: {MIN_YEAR_DEFAULT}, excludes 1940 etc.)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing clean_data files")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else find_data_root()
    out_dir = data_root / CLEAN_SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_root}")
    print(f"Output:    {out_dir}")

    engine = _get_engine()
    if engine is None:
        print("Install netCDF4 or h5netcdf to read .nc files (e.g. pip install netCDF4)")
        return

    nc_by_year, grib_by_year = discover_files(data_root)
    all_years = sorted(set(nc_by_year) | set(grib_by_year))
    if not all_years:
        print("No .nc or .grib files found under data/era5_raw.")
        return
    if args.years:
        all_years = [y for y in all_years if y in args.years]
    else:
        all_years = [y for y in all_years if y >= args.min_year]
    if not all_years:
        print("No years to process (empty list after --min-year filter).")
        return
    print(f"Years to process: {all_years}")

    for year in all_years:
        nc_paths = nc_by_year.get(year, [])
        grib_paths = grib_by_year.get(year, [])

        out_path = out_dir / f"era5_{year}.nc"
        if out_path.exists() and not args.force:
            print(f"\n{year}: {out_path.name} already exists, skipping (use --force to overwrite)")
            continue

        # Temp file: only the final rename creates era5_YYYY.nc, so partial writes are never "complete"
        tmp_path = out_path.parent / (out_path.name + TEMP_SUFFIX)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                print(f"\n{year}: Could not remove stale {tmp_path.name}, skipping")
                continue

        if not nc_paths and not grib_paths:
            continue

        print(f"\n{year}: {len(nc_paths)} .nc, {len(grib_paths)} .grib -> {out_path.name}")

        try:
            if nc_paths:
                ds = combine_nc_for_year(nc_paths, trim=not args.no_trim, engine=engine)
                if not ds.dims or len(ds.data_vars) == 0:
                    print(f"  Skipped (no data).")
                    ds.close()
                    continue
                if grib_paths and not args.nc_only:
                    ds = try_merge_grib(ds, grib_paths)
                ds = drop_object_vars(ds)
                ds.to_netcdf(tmp_path)
                ds.close()
                os.replace(tmp_path, out_path)
                print(f"  Wrote {out_path.name}")
            elif grib_paths and not args.nc_only:
                try:
                    ds = xr.open_mfdataset(
                        [str(p) for p in grib_paths],
                        engine="cfgrib",
                        combine="by_coords",
                    )
                    if "latitude" in ds.dims and "longitude" in ds.dims and not args.no_trim:
                        ds = ds.isel(latitude=LAT_SLICE, longitude=LON_SLICE)
                    ds = drop_object_vars(ds)
                    ds.to_netcdf(tmp_path)
                    ds.close()
                    os.replace(tmp_path, out_path)
                    print(f"  Wrote {out_path.name} (from GRIB)")
                except Exception as e:
                    print(f"  Failed to combine GRIB for {year}: {e}")
                    if tmp_path.exists():
                        tmp_path.unlink()
        except Exception as e:
            print(f"  Error: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    print("\nDone.")


if __name__ == "__main__":
    main()
