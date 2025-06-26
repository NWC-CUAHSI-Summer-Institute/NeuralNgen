import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo import camelsus
from neuralhydrology.utils.config import Config

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)

class HourlyCamelsUS(camelsus.CamelsUS):
    """Data set class providing hourly data for CAMELS US basins.
    
    This class extends the `CamelsUS` dataset class by hourly in- and output data. Currently, only NLDAS forcings are
    available at an hourly resolution.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._netcdf_datasets = {}  # if available, we remember the dataset to load faster
        self._warn_slow_loading = True
        if not any(f.endswith('_hourly') for f in cfg.forcings):
            raise ValueError('Forcings include no hourly forcings set.')
        try:
            LOGGER.debug(f"Initializing HourlyCamelsUS with cfg: {cfg}, is_train: {is_train}, period: {period}, basin: {basin}")
            super(HourlyCamelsUS, self).__init__(cfg=cfg,
                                                 is_train=is_train,
                                                 period=period,
                                                 basin=basin,
                                                 additional_features=additional_features,
                                                 id_to_int=id_to_int,
                                                 scaler=scaler)
            LOGGER.debug("Initialization successful")
        except Exception as e:
            LOGGER.error(f"Error during initialization for basin {basin}: {e}")
            raise

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            if forcing[-7:] == '_hourly':
                df = self.load_hourly_data(basin, forcing)
            else:
                # load daily CAMELS forcings and upsample to hourly
                df, _ = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
                df = df.resample('1H').ffill()
            if len(self.cfg.forcings) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        LOGGER.debug(f"Loaded DataFrame columns: {df.columns.tolist()}")

        # collapse all input features to a single list, to check for 'QObs(mm/d)'.
        all_features = self.cfg.target_variables
        if isinstance(self.cfg.dynamic_inputs, dict):
            for val in self.cfg.dynamic_inputs.values():
                all_features = all_features + val
        elif isinstance(self.cfg.dynamic_inputs, list):
            all_features = all_features + self.cfg.dynamic_inputs

        # catch also QObs(mm/d)_shiftX or _copyX features
        if any([x.startswith("QObs(mm/d)") for x in all_features]):
            if 'QObs(mm/d)' not in df.columns:
                try:
                    # Attempt to load discharge
                    discharge = load_hourly_us_discharge(self.cfg.data_dir, basin)
                    df = df.join(discharge)
                except FileNotFoundError as e:
                    LOGGER.warning(f"Discharge file not found for basin {basin}: {e}. Setting QObs(mm/d) to NaN.")
                    # Create a NaN column for the full time range
                    start_date = pd.to_datetime('1993-09-29 00:00:00')
                    end_date = pd.to_datetime('2013-10-03 23:00:00')
                    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
                    df['QObs(mm/d)'] = pd.Series(index=date_range, dtype=float).reindex(df.index).fillna(np.nan)
                # Fallback using camelsus, with area handling
                _, area = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, "daymet")
                LOGGER.debug(f"Area for basin {basin}: {area}")
                if area is not None and 'QObs(mm/d)' not in df.columns:
                    discharge = camelsus.load_camels_us_discharge(self.cfg.data_dir, basin, area)
                    if discharge is not None:
                        discharge = discharge.resample('1H').ffill()
                        df["QObs(mm/d)"] = discharge
                elif 'QObs(mm/d)' not in df.columns:
                    LOGGER.warning(f"No valid area or discharge data for basin {basin}. Keeping QObs(mm/d) as NaN.")
            else:
                LOGGER.warning(f"'QObs(mm/d)' already exists in forcing data for basin {basin}. Skipping discharge join.")

        # only warn for missing netcdf files once for each forcing product
        self._warn_slow_loading = False

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        # add stage, if requested
        if 'gauge_height_m' in self.cfg.target_variables:
            df = df.join(load_hourly_us_stage(self.cfg.data_dir, basin))
            df.loc[df['gauge_height_m'] < 0, 'gauge_height_m'] = np.nan

        # convert discharge to 'synthetic' stage, if requested
        if 'synthetic_qobs_stage_meters' in self.cfg.target_variables:
            attributes = camelsus.load_camels_us_attributes(data_dir=self.cfg.data_dir, basins=[basin])
            with open(self.cfg.rating_curve_file, 'rb') as f:
                rating_curves = pickle.load(f)
            df['synthetic_qobs_stage_meters'] = np.nan
            if basin in rating_curves.keys():
                discharge_m3s = df['qobs_mm_per_hour'].values / 1000 * attributes.area_gages2[basin] * 1e6 / 60**2
                df['synthetic_qobs_stage_meters'] = rating_curves[basin].discharge_to_stage(discharge_m3s)

        return df

    def load_hourly_data(self, basin: str, forcings: str) -> pd.DataFrame:
        """Load a single set of hourly forcings and discharge. If available, loads from NetCDF, else from csv.
        
        Parameters
        ----------
        basin : str
            Identifier of the basin for which to load data.
        forcings : str
            Name of the forcings set to load.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with forcings and discharge values for the specified basin.
        """
        fallback_csv = False
        try:
            if forcings not in self._netcdf_datasets.keys():
                self._netcdf_datasets[forcings] = load_hourly_us_netcdf(self.cfg.data_dir, forcings)
            df = self._netcdf_datasets[forcings].sel(basin=basin).to_dataframe()
        except FileNotFoundError:
            fallback_csv = True
            if self._warn_slow_loading:
                LOGGER.warning(
                    f'## Warning: Hourly {forcings} NetCDF file not found. Falling back to slower csv files.')
        except KeyError:
            fallback_csv = True
            LOGGER.warning(
                f'## Warning: NetCDF file of {forcings} does not contain data for {basin}. Trying slower csv files.')
        if fallback_csv:
            df = load_hourly_us_forcings(self.cfg.data_dir, basin, forcings)

            # add discharge
            df = df.join(load_hourly_us_discharge(self.cfg.data_dir, basin))

        return df


def load_hourly_us_forcings(data_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load the hourly forcing data for a basin of the CAMELS US data set.

    The hourly forcings are not included in the original data set by Newman et al. (2017).

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain an 'hourly' folder containing one subdirectory
        for each forcing, which contains the forcing files (.csv) for each basin. Files have to contain the 8-digit 
        basin id.
    basin : str
        8-digit USGS identifier of the basin.
    forcings : str
        Must match the folder names in the 'hourly' directory. E.g. 'nldas_hourly'

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    forcing_path = data_dir / 'hourly' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('*.csv'))
    file_path = next((f for f in files if basin in f.stem), None)
    if not file_path:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')

    # Load data and attempt to parse 'time' or 'datetime' as index
    df = pd.read_csv(file_path)
    LOGGER.debug(f"Raw DataFrame columns: {df.columns.tolist()}")
    time_col = 'time'
    if 'time' not in df.columns:
        time_col = 'datetime' if 'datetime' in df.columns else df.columns[0]  # Fallback to first column if no time
        LOGGER.warning(f"'time' column not found, using '{time_col}' as index")
    df = df.set_index(time_col).rename_axis('date')  # Rename index to 'date'
    df.index = pd.to_datetime(df.index)
    LOGGER.debug(f"DataFrame index name after loading: {df.index.name}")
    return df


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly' with a subdirectory 
        'usgs-streamflow' which contains the discharge files (.csv) for each basin. File names must contain the 8-digit 
        basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame of the discharge values (mm/hour)

    Raises
    ------
    FileNotFoundError
        If no discharge file is found for the specified basin.
    """
    pattern = '**/*usgs-hourly.csv'
    discharge_path = data_dir / 'hourly' / 'usgs-streamflow'  # Using hyphen as per your directory
    LOGGER.debug(f"Searching for discharge files in: {discharge_path}")
    files = list(discharge_path.glob(pattern))
    LOGGER.debug(f"Found files: {files}")

    if len(files) == 0:
        discharge_path = discharge_path.parent / 'usgs_streamflow'  # Fallback to underscore
        LOGGER.debug(f"Falling back to: {discharge_path}")
        files = list(discharge_path.glob(pattern))
        LOGGER.debug(f"Found files after fallback: {files}")

    # Try different patterns to match the basin ID
    file_path = next((f for f in files if basin in f.stem), None)
    if not file_path:
        # Try variations if basin ID is not in stem (e.g., prefix or suffix)
        file_path = next((f for f in files if f'{basin}_' in f.name or f'_{basin}' in f.name), None)
    if not file_path:
        LOGGER.warning(f"No discharge file found for basin {basin} at {discharge_path}. Returning empty DataFrame.")
        # Return an empty DataFrame with the correct index range as a fallback
        start_date = pd.to_datetime('1993-09-29 00:00:00')
        end_date = pd.to_datetime('2013-10-03 23:00:00')
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        return pd.DataFrame(index=date_range, columns=['QObs(mm/d)']).fillna(np.nan)
    
    # Load data and attempt to parse 'time' or 'datetime' as index
    df = pd.read_csv(file_path)
    LOGGER.debug(f"Raw Discharge DataFrame columns: {df.columns.tolist()}")
    time_col = 'time'
    if 'time' not in df.columns:
        time_col = 'datetime' if 'datetime' in df.columns else df.columns[0]  # Fallback to first column if no time
        LOGGER.warning(f"'time' column not found in discharge, using '{time_col}' as index")
    df = df.set_index(time_col).rename_axis('date')  # Rename index to 'date'
    df.index = pd.to_datetime(df.index)
    LOGGER.debug(f"Discharge DataFrame index name after loading: {df.index.name}")
    return df


def load_hourly_us_stage(data_dir: Path, basin: str) -> pd.Series:
    """Load the hourly stage data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly' with a subdirectory 
        'usgs_stage' which contains the stage files (.csv) for each basin. File names must contain the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index Series of the stage values (m)
    """
    stage_path = data_dir / 'hourly' / 'usgs_stage'
    files = list(stage_path.glob('**/*_utc.csv'))
    file_path = next((f for f in files if basin in f.stem), None)
    if not file_path:
        raise FileNotFoundError(f'No file for Basin {basin} at {stage_path}')

    df = pd.read_csv(file_path,
                     sep=',',
                     index_col=['datetime'],
                     parse_dates=['datetime'],
                     usecols=['datetime', 'gauge_height_ft'])
    df = df.resample('h').mean()
    df["gauge_height_m"] = df["gauge_height_ft"] * 0.3048

    return df["gauge_height_m"]


def load_hourly_us_netcdf(data_dir: Path, forcings: str) -> xarray.Dataset:
    """Load hourly forcing and discharge data from preprocessed netCDF file.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly', containing the netCDF file.
    forcings : str
        Name of the forcing product. Must match the ending of the netCDF file. E.g. 'nldas_hourly' for 
        'usgs-streamflow-nldas_hourly.nc'

    Returns
    -------
    xarray.Dataset
        Dataset containing the combined discharge and discharge data of all basins (as stored in the netCDF)  
    """
    netcdf_path = data_dir / 'hourly' / f'usgs-streamflow-{forcings}.nc'
    if not netcdf_path.is_file():
        raise FileNotFoundError(f'No NetCDF file for hourly streamflow and {forcings} at {netcdf_path}.')

    return xarray.open_dataset(netcdf_path)

# Main script to load and display data
if __name__ == "__main__":
    data_dir = Path(r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Summer_Institute_clones\CAMELS_data_sample")
    basin = "12035000"
    forcings = "aorc_hourly"
    cfg = Config({
        'data_dir': data_dir,
        'forcings': [forcings],
        'seq_length': 24,
        'predict_last_n': 1,
        'train_start_date': '29/09/1993',
        'train_end_date': '03/10/2013',
        'validation_start_date': '04/10/2013',
        'validation_end_date': '04/10/2013',
        'target_variables': [],
        'dynamic_inputs': ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface', 
                          'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 
                          'VGRD_10maboveground'],
        'loss': 'mse',
        'train_dir': 'runs/train',
        'model': 'lstm',
        'run_dir': 'runs'
    })

    dataset = HourlyCamelsUS(cfg, is_train=True, period='train', basin=basin)
    df = dataset._load_basin_data(basin)

    # Display the first few rows
    print(df.head())

    print("\nFull DataFrame Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())