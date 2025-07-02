# filename: camelsus.py

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from NeuralNGEN.src.dataset.basedataset_NGEN import BaseDataset
from NeuralNGEN.src.utils.utils.config import Config

LOGGER = logging.getLogger(__name__)


class CamelsUS(BaseDataset):
    """Data set class for the daily CAMELS US data set by Newman et al. (2015, 2017).
    
    This class loads daily forcing and discharge data. For hourly data, use a derived class
    like the one in `hourlycamelsus_aorc_Ngen.py`.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded.
    basin : str, optional
        If passed, the data for only this basin will be loaded.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries with additional feature dataframes.
    id_to_int : Dict[str, int], optional
        Mapping from basin string id to integer, required for validation/testing.
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        Dictionary containing normalization statistics, required for validation/testing.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsUS, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load daily input and output data from text files."""
        # Get daily forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df, area = load_camels_us_forcings(self.cfg.data_dir, basin, forcing)

            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # Add daily discharge
        # Note: area is taken from the last forcing file, which is fine as it's the same for all.
        df['QObs(mm/d)'] = load_camels_us_discharge(self.cfg.data_dir, basin, area)

        # Replace invalid discharge values with NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load static attributes."""
        return load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS US attributes from text files."""
    attributes_path = data_dir / 'camels_attributes_v2.0'
    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = list(attributes_path.glob('camels_*.txt'))
    
    dfs = [pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str}).set_index('gauge_id') for f in txt_files]

    df = pd.concat(dfs, axis=1)
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def load_camels_us_forcings(data_dir: Path, basin: str, forcing: str) -> Tuple[pd.DataFrame, int]:
    """Load daily forcing data for a single basin."""
    forcing_path = data_dir / 'basin_mean_forcing' / forcing
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = next(forcing_path.glob(f'**/{basin}_*_forcing_leap.txt'), None)
    if not file_path:
        raise FileNotFoundError(f'No daily forcing file for Basin {basin} at {forcing_path}')

    with open(file_path, 'r') as fp:
        # Load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # Load the dataframe from the rest of the file
        df = pd.read_csv(fp, sep='\s+')
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index("date")

    return df, area


def load_camels_us_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
    """Load daily discharge data for a single basin."""
    discharge_path = data_dir / 'usgs_streamflow'
    file_path = next(discharge_path.glob(f'**/{basin}_streamflow_qc.txt'), None)
    if not file_path:
        raise FileNotFoundError(f'No daily discharge file for Basin {basin} at {discharge_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # Normalize discharge from cfs to mm/day
    df['QObs'] = 28316846.592 * df['QObs'] * 86400 / (area * 10**6)

    return df['QObs']