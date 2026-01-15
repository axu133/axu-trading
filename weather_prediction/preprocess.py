import numpy as np
import pandas as pd

def normalize_data(data, data_min=None, data_max=None):
    """
    Normalize the input data to the range [0, 1].

    :param data: Input numpy array to be normalized.
    :param data_min: Minimum value for normalization. If None, uses the min of the data.
    :param data_max: Maximum value for normalization. If None, uses the max of the data.
    :return: Normalized numpy array.
    """
    if data_min is None:
        data_min = np.min(data)
    if data_max is None:
        data_max = np.max(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized, data_max, data_min

def z_score_normalize(data, mean=None, std=None):
    """
    Apply z-score normalization to the input data.

    :param data: Input numpy array to be normalized.
    :param mean: Mean value for normalization. If None, uses the mean of the data.
    :param std: Standard deviation for normalization. If None, uses the std of the data.
    :return: Z-score normalized numpy array.
    """
    if mean is None:
        mean = np.mean(data, axis=(0, 1, 3, 4), keepdims=True)
    if std is None:
        std = np.std(data, axis=(0, 1, 3, 4), keepdims=True)
    
    z_normalized = (data - mean) / std
    
    return z_normalized, mean, std

def align_dates(era5, labels, target_col = "TMAX", lag_days = 5):
    batch_dates = pd.to_datetime(era5).normalize()

    target_dates = batch_dates + pd.Timedelta(days=lag_days)

    if 'DATE' in labels.columns:
        labels = labels.set_index('DATE')

    labels.index = pd.to_datetime(labels.index)

    aligned = labels[target_col].reindex(target_dates)

    valid_mask = aligned.notna().values

    final_labels = aligned[valid_mask].values

    final_labels = final_labels.reshape(-1, 1).astype(np.float32)

    return valid_mask, final_labels
