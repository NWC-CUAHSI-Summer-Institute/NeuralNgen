# src/neuralngen/dataset/basedataset.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

class BaseDataset(Dataset):
    def __init__(self, cfg, period):
        self.cfg = cfg
        self.period = period

        # load basins
        basin_file = getattr(cfg, f"{period}_basin_file")
        with open(basin_file, 'r') as f:
            self.basins = [line.strip() for line in f.readlines()]

        # load attributes
        static_attr_df = self._load_static_attributes()

        self.static_attributes = {}
        self.basin_info = {}

        for basin in self.basins:
            # ----
            # Build static feature vector (x_s)
            # ----
            attr_names = self.cfg.static_attributes

            # Check attributes exist
            for attr in attr_names:
                if attr not in static_attr_df.columns:
                    raise ValueError(f"Static attribute {attr} missing from attribute file.")

            numeric_values = static_attr_df.loc[basin, attr_names].values.astype(np.float32)
            self.static_attributes[basin] = numeric_values

            # ----
            # Build basin_info dictionary
            # ----
            lat = static_attr_df.loc[basin, 'gauge_lat'] if 'gauge_lat' in static_attr_df.columns else np.nan
            lon = static_attr_df.loc[basin, 'gauge_lon'] if 'gauge_lon' in static_attr_df.columns else np.nan

            self.basin_info[basin] = {
                "gauge_id": basin,
                "gauge_lat": float(lat),
                "gauge_lon": float(lon),
            }

        # load all time series
        self.dynamic_inputs = {}
        self.targets = {}

        for basin in self.basins:
            df = self._load_basin_data(basin)

            start_date = pd.to_datetime(getattr(cfg, f"{period}_start_date"))
            end_date = pd.to_datetime(getattr(cfg, f"{period}_end_date"))

            df = df.loc[start_date:end_date]

            # drop rows with NaNs in dynamic inputs or targets
            df = df.dropna(subset=cfg.dynamic_inputs + cfg.target_variables)

            self.dynamic_inputs[basin] = df[cfg.dynamic_inputs].values.astype(np.float32)
            self.targets[basin] = df[cfg.target_variables].values.astype(np.float32)

        # --------------------------------
        # Build sample list for all windows
        # --------------------------------
        self.samples = []
        self.window_size = cfg.batch_timesteps + cfg.sequence_length

        for basin in self.basins:
            dyn = self.dynamic_inputs[basin]
            if dyn.shape[0] < self.window_size:
                continue

            n_windows = dyn.shape[0] - self.window_size + 1

            for start_idx in range(n_windows):
                self.samples.append((basin, start_idx))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basin, start_idx = self.samples[idx]

        end_idx = start_idx + self.window_size

        x_d_window = self.dynamic_inputs[basin][start_idx:end_idx, :]
        y_window = self.targets[basin][start_idx:end_idx, :]

        x_s = self.static_attributes[basin]
        x_info = self.basin_info[basin]

        return {
            "x_d": torch.tensor(x_d_window, dtype=torch.float32),
            "x_s": torch.tensor(x_s, dtype=torch.float32),
            "y": torch.tensor(y_window, dtype=torch.float32),
            "x_info": x_info,
        }

    def _load_static_attributes(self):
        raise NotImplementedError()

    def _load_basin_data(self, basin):
        raise NotImplementedError()