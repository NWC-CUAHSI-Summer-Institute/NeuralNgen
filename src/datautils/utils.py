import functools
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray
from pandas.tseries.frequencies import to_offset
from ruamel.yaml import YAML
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Pandas switched from "Y" to "YE" and similar identifiers in 2.2.0. This snippet checks which one is correct for the
# current pandas installation.
_YE_FREQ = 'YE'
_ME_FREQ = 'ME'
_QE_FREQ = 'QE'
try:
    to_offset(_YE_FREQ)
except ValueError:
    _YE_FREQ = 'Y'
    _ME_FREQ = 'M'
    _QE_FREQ = 'Q'


def load_scaler(run_dir: Path) -> Dict[str, Union[pd.Series, xarray.Dataset]]:
    """Load feature scaler from run directory.

    Checks run directory for scaler file in yaml format (new) or pickle format (old).

    Parameters
    ----------
    run_dir: Path
        Run directory. Has to contain a folder 'train_data' that contains the 'train_data_scaler' file.

    Returns
    -------
    Dictionary, containing the feature scaler for static and dynamic features.
    
    Raises
    ------
    FileNotFoundError
        If neither a 'train_data_scaler.yml' or 'train_data_scaler.p' file is found in the 'train_data' folder of the 
        run directory.
    """
    scaler_file = run_dir / "train_data" / "train_data_scaler.yml"

    if scaler_file.is_file():
        # read scaler from disk
        with scaler_file.open("r") as fp:
            yaml = YAML(typ="safe")
            scaler_dump = yaml.load(fp)

        # transform scaler into the format expected by NeuralHydrology
        scaler = {}
        for key, value in scaler_dump.items():
            if key in ["attribute_means", "attribute_stds", "camels_attr_means", "camels_attr_stds"]:
                scaler[key] = pd.Series(value)
            elif key in ["xarray_feature_scale", "xarray_feature_center"]:
                scaler[key] = xarray.Dataset.from_dict(value).astype(np.float32)

        return scaler

    else:
        scaler_file = run_dir / "train_data" / "train_data_scaler.p"

        if scaler_file.is_file():
            with scaler_file.open('rb') as fp:
                scaler = pickle.load(fp)
            return scaler
        else:
            raise FileNotFoundError(f"No scaler file found in {scaler_file.parent}. "
                                    "Looked for (new) yaml file or (old) pickle file")


def load_hydroatlas_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load HydroATLAS attributes into a pandas DataFrame

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of the dataset. Must contain a folder called 'hydroatlas_attributes' with a file
        called `attributes.csv`. The attributes file is expected to have one column called `basin_id`.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame containing the HydroATLAS attributes.
    """
    attribute_file = data_dir / "hydroatlas_attributes" / "attributes.csv"
    if not attribute_file.is_file():
        raise FileNotFoundError(attribute_file)

    df = pd.read_csv(attribute_file, dtype={'basin_id': str})
    df = df.set_index('basin_id')

    if basins:
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df


def load_basin_file(basin_file: Path) -> List[str]:
    """Load list of basins from text file.
    
    Note: Basins names are not allowed to end with '_period*'
    
    Parameters
    ----------
    basin_file : Path
        Path to a basin txt file. File has to contain one basin id per row, while empty rows are ignored.

    Returns
    -------
    List[str]
        List of basin ids as strings.
        
    Raises
    ------
    ValueError
        In case of invalid basin names that would cause problems internally.
    """
    with basin_file.open('r') as fp:
        basins = sorted(basin.strip() for basin in fp if basin.strip())

    # sanity check basin names
    problematic_basins = [basin for basin in basins if basin.split('_')[-1].startswith('period')]
    if problematic_basins:
        msg = [
            f"The following basin names are invalid {problematic_basins}. Check documentation of the ",
            "'load_basin_file()' functions for details."
        ]
        raise ValueError(" ".join(msg))

    return basins


def attributes_sanity_check(df: pd.DataFrame):
    """Utility function to check the suitability of the attributes for model training.
    
    This utility function can be used to check if any attribute has a standard deviation of zero. This would lead to 
    NaN's when normalizing the features and thus would lead to NaN's when training the model. It also checks if any
    attribute for any basin contains a NaN, which would also cause NaNs during model training.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of catchment attributes as columns.

    Raises
    ------
    RuntimeError
        If one or more attributes have a standard deviation of zero or any attribute for any basin is NaN.
    """
    # Check for NaNs in standard deviation of attributes.
    attributes = []
    if any(df.std() == 0.0) or any(df.std().isnull()):
        for k, v in df.std().items():
            if (v == 0) or (np.isnan(v)):
                attributes.append(k)
    if attributes:
        msg = [
            "The following attributes have a std of zero or NaN, which results in NaN's ",
            "when normalizing the features. Remove the attributes from the attribute feature list ",
            "and restart the run. \n", f"Attributes: {attributes}"
        ]
        raise RuntimeError("".join(msg))

    # Check for NaNs in any attribute of any basin
    nan_df = df[df.isnull().any(axis=1)]
    if len(nan_df) > 0:
        failure_cases = defaultdict(list)
        for basin, row in nan_df.iterrows():
            for feature, value in row.items():
                if np.isnan(value):
                    failure_cases[basin].append(feature)
        # create verbose error message
        msg = ["The following basins/attributes are NaN, which can't be used as input:"]
        for basin, features in failure_cases.items():
            msg.append(f"{basin}: {features}")
        raise RuntimeError("\n".join(msg))



def infer_datetime_coord(xr: Union[DataArray, Dataset]) -> str:
    """Checks for coordinate with 'date' in its name and returns the name.
    
    Parameters
    ----------
    xr : Union[DataArray, Dataset]
        Array to infer coordinate name of.
        
    Returns
    -------
    str
        Name of datetime coordinate name.
        
    Raises
    ------
    RuntimeError
        If none or multiple coordinates with 'date' in its name are found.
    """
    candidates = [c for c in list(xr.coords) if "date" in c]
    if len(candidates) > 1:
        raise RuntimeError("Found multiple coordinates with 'date' in its name.")
    if not candidates:
        raise RuntimeError("Did not find any coordinate with 'date' in its name")

    return candidates[0]


