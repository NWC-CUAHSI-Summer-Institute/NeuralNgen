# filename: hourlycamelsus_aorc_Ngen.py

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray
import traceback  # <--- ADD THIS LINE

from NeuralNGEN.src.dataset import camelsus_NGEN
from NeuralNGEN.src.utils.config import Config

LOGGER = logging.getLogger(__name__)


class HourlyCamelsUS(camelsus_NGEN.CamelsUS):
    """
    Data set class providing hourly data for CAMELS US basins.
    
    This class extends the daily `camelsus.CamelsUS` dataset by adding support for
    hourly forcings and observations.
    """
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._netcdf_datasets = {}
        self._warn_slow_loading = True
        if not any(f.endswith('_hourly') for f in cfg.forcings):
            LOGGER.warning('No hourly forcing sets specified in config. Falling back to daily data.')
        
        super(HourlyCamelsUS, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load hourly input and output data from files."""
        # Load hourly forcings specified in the config
        dfs = []
        for forcing in self.cfg.forcings:
            if forcing.endswith('_hourly'):
                df_forcing = self.load_hourly_data(basin, forcing)
            else:
                df_forcing, _ = camelsus_NGEN.load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
                df_forcing = df_forcing.resample('h').ffill() # Use 'h' for hourly frequency
            
            if len(self.cfg.forcings) > 1:
                df_forcing = df_forcing.rename(columns={col: f"{col}_{forcing}" for col in df_forcing.columns})
            dfs.append(df_forcing)
        
        df = pd.concat(dfs, axis=1)

        if any(target.startswith("QObs") for target in self.cfg.target_variables):
            try:
                discharge_hourly = load_hourly_us_discharge(self.cfg.data_dir, basin)
                df = df.join(discharge_hourly, how='left')
            except FileNotFoundError:
                LOGGER.warning(f"No hourly discharge file found for basin {basin}. Target values will be NaN.")
                df["QObs(mm/d)"] = np.nan

        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan
            
        return df

    def load_hourly_data(self, basin: str, forcing: str) -> pd.DataFrame:
        """Load a single set of hourly forcings, trying NetCDF then falling back to CSV."""
        try:
            if forcing not in self._netcdf_datasets:
                self._netcdf_datasets[forcing] = load_hourly_us_netcdf(self.cfg.data_dir, forcing)
            df = self._netcdf_datasets[forcing].sel(basin=basin).to_dataframe()
            return df
        except (FileNotFoundError, KeyError):
            LOGGER.warning(f'Hourly NetCDF for {forcing} not found or basin {basin} is missing. Falling back to CSVs.')
            self._warn_slow_loading = False
            return load_hourly_us_forcings(self.cfg.data_dir, basin, forcing)


def find_time_column(df: pd.DataFrame) -> str:
    """Find the time column in a DataFrame, checking for common names."""
    for col_name in ['date', 'datetime', 'time']:
        if col_name in df.columns:
            return col_name
    raise ValueError(f"No suitable time column found. Expected 'date', 'datetime', or 'time'. Found: {df.columns.tolist()}")


def load_hourly_us_forcings(data_dir: Path, basin: str, forcing: str) -> pd.DataFrame:
    """Load hourly forcing data for a basin from a .csv file."""
    forcing_path = data_dir / 'hourly' / forcing
    if not forcing_path.is_dir():
        raise OSError(f"Forcing directory not found: {forcing_path}")

    file_path = next(forcing_path.glob(f'*{basin}*.csv'), None)
    if not file_path:
        raise FileNotFoundError(f'No hourly forcing file for basin {basin} at {forcing_path}')

    df = pd.read_csv(file_path)
    time_col = find_time_column(df)
    df = df.set_index(pd.to_datetime(df[time_col])).rename_axis('date')
    return df.drop(columns=[time_col], errors='ignore')


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load hourly discharge data for a basin from a .csv file."""
    for dirname in ['usgs-streamflow', 'usgs_streamflow']:
        discharge_path = data_dir / 'hourly' / dirname
        if discharge_path.is_dir():
            file_path = next(discharge_path.glob(f'**/*{basin}*usgs-hourly.csv'), None)
            if file_path:
                df = pd.read_csv(file_path)
                time_col = find_time_column(df)
                df = df.set_index(pd.to_datetime(df[time_col])).rename_axis('date')
                return df.drop(columns=[time_col], errors='ignore')

    raise FileNotFoundError(f'No hourly discharge file found for basin {basin}')


def load_hourly_us_netcdf(data_dir: Path, forcing: str) -> xarray.Dataset:
    """Load hourly data from a preprocessed NetCDF file."""
    netcdf_path = data_dir / 'hourly' / f'usgs-streamflow-{forcing}.nc'
    if not netcdf_path.is_file():
        raise FileNotFoundError(f'No NetCDF file at {netcdf_path}.')
    return xarray.open_dataset(netcdf_path)

