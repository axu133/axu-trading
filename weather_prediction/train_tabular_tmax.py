import argparse
from dataclasses import dataclass
import math
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

try:
    # Preferred when running: `python -m weather_prediction.train_tabular_tmax`
    from .read_data import ERA5Dataset
except ImportError as e:  # pragma: no cover
    # Only fall back when we're missing package context (e.g. running as a script).
    # Do NOT hide real dependency import errors from inside weather_prediction.
    if "relative import" in str(e) or "no known parent package" in str(e):
        from weather_prediction.read_data import ERA5Dataset
    else:
        raise


@dataclass(frozen=True)
class SplitConfig:
    train_years: range
    val_years: range
    test_years: range


def _split_indices_by_target_year(dataset: ERA5Dataset, split: SplitConfig):
    idx_train, idx_val, idx_test = [], [], []
    for idx in range(len(dataset)):
        _, _, _, _, target_year = dataset.valid_samples[idx]
        if target_year in split.train_years:
            idx_train.append(idx)
        elif target_year in split.val_years:
            idx_val.append(idx)
        elif target_year in split.test_years:
            idx_test.append(idx)
    return idx_train, idx_val, idx_test


def _fit_climatology_by_doy(dataset: ERA5Dataset, indices):
    clim_sum = np.zeros(367, dtype=np.float64)
    clim_cnt = np.zeros(367, dtype=np.int64)
    for idx in indices:
        _, y_abs, _, doy, _ = dataset[idx]
        d = int(doy)
        if 1 <= d <= 366:
            clim_sum[d] += float(y_abs)
            clim_cnt[d] += 1
    clim = np.full(367, np.nan, dtype=np.float32)
    ok = clim_cnt > 0
    clim[ok] = (clim_sum[ok] / clim_cnt[ok]).astype(np.float32)
    return clim


def _doy_sin_cos(doy: int):
    # Use 365.25 to smooth leap years a bit.
    ang = 2.0 * math.pi * (float(doy) / 365.25)
    return math.sin(ang), math.cos(ang)


def _extract_features(x_ch_t_hw: np.ndarray):
    """
    x: float array shaped [C, T, H, W] (typically normalized already).
    Returns 1D feature vector.
    """
    C = x_ch_t_hw.shape[0]
    feats = []
    for c in range(C):
        v = x_ch_t_hw[c]
        feats.extend(
            [
                float(np.nanmean(v)),
                float(np.nanstd(v)),
                float(np.nanmin(v)),
                float(np.nanmax(v)),
            ]
        )

    # Derived: wind speed (requires u10,v10 at channels 1,2)
    if C >= 3:
        u = x_ch_t_hw[1]
        v = x_ch_t_hw[2]
        wspd = np.sqrt(u * u + v * v)
        feats.extend(
            [
                float(np.nanmean(wspd)),
                float(np.nanstd(wspd)),
                float(np.nanmax(wspd)),
            ]
        )

    # Derived: dewpoint depression (t2m - d2m, channels 0 and 4 in current code)
    if C >= 5:
        dd = x_ch_t_hw[0] - x_ch_t_hw[4]
        feats.extend(
            [
                float(np.nanmean(dd)),
                float(np.nanstd(dd)),
                float(np.nanmin(dd)),
                float(np.nanmax(dd)),
            ]
        )

    return np.asarray(feats, dtype=np.float32)


def _build_xy(dataset: ERA5Dataset, indices, clim_mean_by_doy: np.ndarray, max_samples: int | None = None):
    if max_samples is not None:
        indices = indices[:max_samples]

    X_list = []
    y_list = []
    y_persist_list = []
    y_climo_list = []

    for idx in indices:
        x, y_abs, y_persist, doy, _ = dataset[idx]
        x_np = x.numpy()  # [C, T, H, W]

        feats = _extract_features(x_np)
        doy_i = int(doy)
        s, c = _doy_sin_cos(doy_i)

        # Add calendar + strong scalar baselines as features
        climo = float(clim_mean_by_doy[doy_i]) if 1 <= doy_i <= 366 else float("nan")
        feats = np.concatenate(
            [
                feats,
                np.asarray([float(doy_i), s, c, float(y_persist), climo], dtype=np.float32),
            ]
        )

        X_list.append(feats)
        y_list.append(float(y_abs))
        y_persist_list.append(float(y_persist))
        y_climo_list.append(climo)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.float32)
    y_persist = np.asarray(y_persist_list, dtype=np.float32)
    y_climo = np.asarray(y_climo_list, dtype=np.float32)
    return X, y, y_persist, y_climo


def main():
    ap = argparse.ArgumentParser(description="Tabular baseline for next-day Central Park TMAX (°F).")
    ap.add_argument("--min-year", type=int, default=1980)
    ap.add_argument("--max-year", type=int, default=2025)
    ap.add_argument("--train-end-year", type=int, default=2017, help="Last target year included in train.")
    ap.add_argument("--val-end-year", type=int, default=2021, help="Last target year included in val.")
    ap.add_argument("--test-end-year", type=int, default=2025, help="Last target year included in test.")
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for quick runs.")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--l2-regularization", type=float, default=0.0)
    args = ap.parse_args()

    years = range(args.min_year, args.max_year + 1)
    split = SplitConfig(
        train_years=range(args.min_year, min(args.train_end_year, args.max_year) + 1),
        val_years=range(max(args.train_end_year + 1, args.min_year), min(args.val_end_year, args.max_year) + 1),
        test_years=range(max(args.val_end_year + 1, args.min_year), min(args.test_end_year, args.max_year) + 1),
    )

    ds = ERA5Dataset(years=years, window_size=args.window_size, max_or_min="max", normalize=False)

    # Fit normalization on training years only, then apply globally (no leakage).
    mean, std = ds.fit_normalization_for_years(split.train_years)
    ds.apply_normalization(mean=mean, std=std)

    idx_train, idx_val, idx_test = _split_indices_by_target_year(ds, split)
    if not idx_train or not idx_val or not idx_test:
        raise RuntimeError(
            f"Empty split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}. "
            "Check year ranges and data availability."
        )

    clim = _fit_climatology_by_doy(ds, idx_train)

    X_train, y_train, y_train_p, y_train_c = _build_xy(ds, idx_train, clim, max_samples=args.max_samples)
    X_val, y_val, y_val_p, y_val_c = _build_xy(ds, idx_val, clim, max_samples=args.max_samples)
    X_test, y_test, y_test_p, y_test_c = _build_xy(ds, idx_test, clim, max_samples=args.max_samples)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=1,
    )
    model.fit(X_train, y_train)

    def report(split_name, X, y, y_persist, y_climo):
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        mae_persist = mean_absolute_error(y, y_persist)
        ok = np.isfinite(y_climo)
        mae_climo = mean_absolute_error(y[ok], y_climo[ok]) if np.any(ok) else float("nan")
        print(
            f"{split_name:<5} | "
            f"MAE_abs={mae:6.3f} | "
            f"MAE_persist={mae_persist:6.3f} | "
            f"MAE_climo={mae_climo:6.3f} | "
            f"N={len(y)}"
        )

    print("Tabular baseline (HistGradientBoostingRegressor)")
    report("train", X_train, y_train, y_train_p, y_train_c)
    report("val", X_val, y_val, y_val_p, y_val_c)
    report("test", X_test, y_test, y_test_p, y_test_c)


if __name__ == "__main__":
    main()

