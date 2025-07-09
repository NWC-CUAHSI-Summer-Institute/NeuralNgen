# ./src/neuralngen/dataset/normalization.py

import numpy as np
import pandas as pd
import json
from pathlib import Path

def convert_cms_to_mmhr(q_cms: np.ndarray, area_km2: float) -> np.ndarray:
    """
    Convert discharge from m³/s to mm/hr.

    Parameters
    ----------
    q_cms : np.ndarray
        Array of discharge values in m³/s.
    area_km2 : float
        Basin area in square kilometers.

    Returns
    -------
    np.ndarray
        Discharge values converted to mm/hr.
    """
    return (q_cms * 1e3) / (area_km2 * 3600.0)

def compute_scalers(dataframes, variables):
    """
    Compute mean/std scalers for a list of DataFrames.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Each DataFrame corresponds to a basin.
    variables : list of str
        Column names to scale.

    Returns
    -------
    dict
        Mapping {var: {mean, std}}
    """
    all_data = []
    for df in dataframes:
        all_data.append(df[variables].values)
    stacked = np.vstack(all_data)

    scalers = {}
    for i, var in enumerate(variables):
        mean = np.nanmean(stacked[:, i])
        std = np.nanstd(stacked[:, i])
        scalers[var] = {"mean": float(mean), "std": float(std)}
    return scalers

def apply_scalers(df, scalers):
    """
    Apply mean/std scaling to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scale in-place.
    scalers : dict
        e.g. {var: {mean, std}}

    Returns
    -------
    pd.DataFrame
        Scaled dataframe.
    """
    df_scaled = df.copy()
    for var, stats in scalers.items():
        mean = stats["mean"]
        std = stats["std"]
        if std > 0:
            df_scaled[var] = (df_scaled[var] - mean) / std
        else:
            df_scaled[var] = df_scaled[var] - mean
    return df_scaled


def save_scalers(scalers, filepath):
    with open(filepath, 'w') as f:
        json.dump(scalers, f, indent=4)

def load_scalers(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def inverse_scale(scaled_df, scalers):
    """
    Inverse transform scaled data back to original units.
    """
    unscaled_df = scaled_df.copy()
    for col in scaled_df.columns:
        if col in scalers:
            mean = scalers[col]["mean"]
            std = scalers[col]["std"]
            unscaled_df[col] = scaled_df[col] * std + mean
    return unscaled_df

def runoff_mmhr_to_cms(runoff_mmhr, basin_area_km2):
    """
    Convert runoff in mm/hr to discharge in m³/s.
    """
    return (runoff_mmhr * basin_area_km2 * 3600) / 1e3

def runoff_mmhr_to_cfs(runoff_mmhr, basin_area_km2):
    """
    Convert runoff in mm/hr to discharge in ft³/s.
    """
    cms = runoff_mmhr_to_cms(runoff_mmhr, basin_area_km2)
    return cms * 35.3147