# src/neuralngen/dataset/parallel_load.py

import pandas as pd
import numpy as np
from neuralngen.dataset.hourlycamels import HourlyCamelsDataset

def load_basin_for_parallel(cfg_dict, period, basin):
    """
    Loads data for a single basin for use in parallel processing.

    Parameters
    ----------
    cfg_dict : dict
        A dictionary of config values (because Config objects are not picklable).
    period : str
        Train, validation, or test.
    basin : str
        Basin ID.

    Returns
    -------
    basin : str
        The basin ID.
    dynamic_inputs : np.ndarray
        Dynamic input array [T, Din].
    targets : np.ndarray
        Target array [T, Dout].
    """
    # Reconstruct Config object from dictionary
    from neuralngen.utils import Config
    cfg = Config(cfg_dict)

    # Create a temporary dataset instance
    ds = HourlyCamelsDataset(cfg, is_train=(period=="train"), period=period)
    df = ds._load_basin_data(basin)

    # Restrict dates
    start_date = pd.to_datetime(getattr(cfg, f"{period}_start_date"))
    end_date = pd.to_datetime(getattr(cfg, f"{period}_end_date"))
    df = df.loc[start_date:end_date]

    # Drop missing rows
    df = df.dropna(subset=cfg.dynamic_inputs + cfg.target_variables)

    dyn = df[cfg.dynamic_inputs].values.astype(np.float32)
    tgt = df[cfg.target_variables].values.astype(np.float32)

    return basin, dyn, tgt
