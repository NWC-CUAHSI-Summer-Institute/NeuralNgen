
# filename: basedataset.py

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union
import sys
import numpy as np
import pandas as pd
import torch
import xarray
from numba import njit, prange
from pandas.tseries.frequencies import to_offset
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm

from NeuralNGEN.src.datautils import utils
from NeuralNGEN.src.utils.config import Config
from NeuralNGEN.src.utils.errors import NoEvaluationDataError, NoTrainDataError

LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base data set class to load and preprocess data, simplified for hourly use.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.period = period
        self.basins = basin if basin else utils.load_basin_file(getattr(cfg, f"{period}_basin_file"))
        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler

        self._compute_scaler = is_train and not scaler
        # FIX: Use getattr for safe access to the 'verbose' attribute
        self._disable_pbar = getattr(cfg, 'verbose', 1) == 0

        # Simplified frequency handling for single-frequency (hourly) data
        self.seq_length = self.cfg.seq_length
        self.predict_last_n = self.cfg.predict_last_n
        
        # Initialize class attributes
        self._x_d = {}
        self._attributes = {}
        self._y = {}
        self._per_basin_target_stds = {}
        self._dates = {}
        self.num_samples = 0
        self.lookup_table = {}

        self._load_data()

        if self.is_train:
            self._dump_scaler()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int) -> dict:
        basin, idx = self.lookup_table[item]
        start_idx = idx + 1 - self.seq_length

        sample = {
            'x_d': self._x_d[basin][start_idx:idx + 1],
            'y': self._y[basin][start_idx:idx + 1],
            'date': self._dates[basin][start_idx:idx + 1]
        }
        
        if self._attributes:
            sample['x_s'] = self._attributes[basin]
        
        scoring_attr = {'gauge_id': basin,
                'lat': self._loss_attrs['lat'][basin],
                'lon': self._loss_attrs['lon'][basin],
                'Clusters': self._loss_attrs['Clusters'][basin]}
        
        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]

        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(
                torch.tensor(self.id_to_int[basin]),
                num_classes=len(self.id_to_int)
            ).to(torch.float32)

        return [sample,scoring_attr]

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function must be implemented by a subclass."""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """This function must be implemented by a subclass."""
        raise NotImplementedError

    def _create_id_to_int(self):
        self.id_to_int = {str(b): i for i, b in enumerate(np.random.permutation(self.basins))}
        file_path = Path(self.cfg.train_dir) / "id_to_int.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(self.id_to_int, fp)

    def _dump_scaler(self):
        scaler_dict = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, (pd.Series, xarray.DataArray, xarray.Dataset)):
                scaler_dict[key] = value.to_dict()
        file_path = Path(self.cfg.train_dir) / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler_dict), fp)
            
    def _get_start_and_end_dates(self, basin: str) -> tuple:
        start_date = getattr(self.cfg, f"{self.period}_start_date")
        end_date = getattr(self.cfg, f"{self.period}_end_date")
        return start_date, end_date

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        data_list = []
        keep_cols = self.cfg.target_variables + self.cfg.dynamic_inputs

        LOGGER.info("Loading basin data into xarray data set.")
        for basin in tqdm(self.basins, disable=self._disable_pbar):
            df = self._load_basin_data(basin)
            
            # Add features from any additionally passed files
            if self.additional_features:
                df = pd.concat([df, *[d[basin] for d in self.additional_features]], axis=1)

            # During evaluation, add missing target columns as NaNs
            if not self.is_train:
                for var in self.cfg.target_variables:
                    if var not in df.columns:
                        df[var] = np.nan
            
            df = df[[col for col in keep_cols if col in df.columns]]
            
            start_date, end_date = self._get_start_and_end_dates(basin)
            
            # The data is hourly, so frequency is '1H'
            warmup = to_offset('1H') * (self.seq_length - self.predict_last_n)
            
            df_period = df[start_date - warmup : end_date]
            
            # Set targets before the official start date to NaN to prevent using them in warmup
            df_period.loc[df_period.index < start_date, self.cfg.target_variables] = np.nan
            
            xr_basin = xarray.Dataset.from_dataframe(df_period.astype(np.float32))
            xr_basin = xr_basin.assign_coords({'basin': basin})
            data_list.append(xr_basin)

        if not data_list:
            raise NoTrainDataError if self.is_train else NoEvaluationDataError

        return xarray.concat(data_list, dim="basin")
    
    def _calculate_per_basin_std(self, xr: xarray.Dataset):
        LOGGER.info("Calculating target variable stds per basin")
        for basin in tqdm(self.basins, disable=self._disable_pbar):
            obs = xr.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                self._per_basin_target_stds[basin] = torch.tensor(np.nanstd(obs, axis=1), dtype=torch.float32)
            else:
                self._per_basin_target_stds[basin] = torch.full((obs.shape[0],), np.nan, dtype=torch.float32)

    def _create_lookup_table(self, xr: xarray.Dataset):
        LOGGER.info("Creating lookup table and converting to pytorch tensors.")
        
        for basin in tqdm(xr["basin"].values, disable=self._disable_pbar):
            df_basin = xr.sel(basin=basin).to_dataframe()
            
            y_array = df_basin[self.cfg.target_variables].values
            valid_indices = _validate_samples(y_array, self.seq_length, self.predict_last_n)
            valid_indices_flat = np.argwhere(valid_indices == 1).flatten()
            
            if len(valid_indices_flat) > 0:
                for index in valid_indices_flat:
                    self.lookup_table[len(self.lookup_table)] = (basin, index)
                
                # self._x_d[basin] = {col: torch.from_numpy(df_basin[[col]].values.astype(np.float32)) 
                #                     for col in self.cfg.dynamic_inputs}
                x_d_array = df_basin[self.cfg.dynamic_inputs].values.astype(np.float32)
                self._x_d[basin] = torch.from_numpy(x_d_array)
                self._y[basin] = torch.from_numpy(y_array.astype(np.float32))
                self._dates[basin] = df_basin.index.to_numpy()

        self.num_samples = len(self.lookup_table)
        if self.num_samples == 0:
            raise NoTrainDataError if self.is_train else NoEvaluationDataError

    def _load_combined_attributes(self):
        # FIX: Use getattr for safe access to optional 'static_attributes' config
        if not getattr(self.cfg, 'static_attributes', None):
            return

        df = self._load_attributes()
        ## retun rais error if lat,lon, cluster not in df
        required_attrs = ['lat', 'lon', 'Clusters']
        missing_attrs = [attr for attr in required_attrs if attr not in df.columns]

        if missing_attrs:
            print(f"WARNING: Missing static attributes: {missing_attrs} in CAMEL attribute input files")
            # self.cfg.static_attributes.extend(missing_attrs)
        loss_attrs =   df[required_attrs]
        loss_attrs = loss_attrs.to_dict()
        self._loss_attrs = loss_attrs
        df = df[self.cfg.static_attributes]
        
        if self._compute_scaler:
            self.scaler["attribute_means"] = df.mean()
            self.scaler["attribute_stds"] = df.std()

        # Normalize only selected columns
        df_normalized = (df - self.scaler["attribute_means"]) / self.scaler["attribute_stds"]

        # Optionally: you can also encode or leave 'Clusters' as categorical if needed

        # Save final attributes
        for basin in self.basins:
            if basin in df_normalized.index:
                self._attributes[basin] = torch.from_numpy(df_normalized.loc[basin].values.astype(np.float32))

    def _load_data(self):
        # FIX: Use getattr for safe access to optional 'use_basin_id_encoding' config
        if getattr(self.cfg, 'use_basin_id_encoding', False) and self.is_train:
            self._create_id_to_int()

        self._load_combined_attributes()
        xr_data = self._load_or_create_xarray_dataset()
        
        if hasattr(self.cfg, 'loss') and str(self.cfg.loss).lower() in ['nse', 'weightednse']:
            self._calculate_per_basin_std(xr_data)
        
        if self._compute_scaler:
            self.scaler["xarray_feature_center"] = xr_data.mean(skipna=True)
            self.scaler["xarray_feature_scale"] = xr_data.std(skipna=True)

        xr_normalized = (xr_data - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]
        self._create_lookup_table(xr_normalized)
        

@njit
def _validate_samples(y: np.ndarray, seq_length: int, predict_last_n: int) -> np.ndarray:
    """Checks for invalid samples due to NaN or insufficient sequence length."""
    n_samples = len(y)
    flags = np.ones(n_samples, dtype=np.int32)
    
    for j in prange(n_samples):
        if j < seq_length - 1:
            flags[j] = 0
            continue
            
        target_slice = y[j - predict_last_n + 1 : j + 1]
        if np.all(np.isnan(target_slice)):
            flags[j] = 0
            
    return flags

