# src/neuralngen/dataset/basedataset.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, cfg, period):
        self.cfg = cfg
        self.period = period

        # load basin IDs
        basin_file = getattr(cfg, f"{period}_basin_file")
        with open(basin_file, 'r') as f:
            self.basins = [line.strip() for line in f.readlines()]

        # load static attributes
        static_attr_df = self._load_static_attributes()

        # filter numeric columns once
        numeric_df = static_attr_df.select_dtypes(include=[np.number])

        # define the attribute first!
        self.static_attributes = {}

        for basin in self.basins:
            if basin not in numeric_df.index:
                raise RuntimeError(f"Basin {basin} not found in attributes table.")

            numeric_values = numeric_df.loc[basin].values
            self.static_attributes[basin] = numeric_values.astype(np.float32)

        # load timeseries data
        self.dynamic_inputs = {}
        self.targets = {}

        for basin in self.basins:
            df = self._load_basin_data(basin)

            # slice to desired period
            start_date = pd.to_datetime(getattr(cfg, f"{period}_start_date"), dayfirst=True)
            end_date = pd.to_datetime(getattr(cfg, f"{period}_end_date"), dayfirst=True)

            df = df.loc[start_date:end_date]

            # separate inputs and targets
            self.dynamic_inputs[basin] = df[cfg.dynamic_inputs].values.astype(np.float32)
            self.targets[basin] = df[cfg.target_variables].values.astype(np.float32)

            # optionally normalize here if desired

        self.basin_ids = list(self.dynamic_inputs.keys())

    def _load_static_attributes(self):
        # same as camelsus.load_camels_us_attributes(), simplified
        # Load and return a DataFrame indexed by basin IDs
        # For now, e.g.:
        #   index = basin ID
        #   columns = static attributes
        raise NotImplementedError()

    def _load_basin_data(self, basin):
        # load hourly CAMELS data
        raise NotImplementedError()

    def __len__(self):
        # each item returns a full batch!
        return 1000000   # arbitrary large number, sampling randomly each time

    def __getitem__(self, idx):
        cfg = self.cfg
        batch_sites = cfg.batch_sites
        batch_timesteps = cfg.batch_timesteps

        chosen_basins = np.random.choice(self.basin_ids, size=batch_sites, replace=True)

        batch_x_d = []
        batch_x_s = []
        batch_y = []

        for basin in chosen_basins:
            dyn = self.dynamic_inputs[basin]
            y = self.targets[basin]

            if dyn.shape[0] < batch_timesteps:
                raise RuntimeError(f"Basin {basin} has fewer timesteps than batch_timesteps.")

            # random window
            start_idx = np.random.randint(0, dyn.shape[0] - batch_timesteps + 1)
            end_idx = start_idx + batch_timesteps

            x_d_window = dyn[start_idx:end_idx, :]
            y_window = y[start_idx:end_idx, :]

            x_s = self.static_attributes[basin]

            batch_x_d.append(torch.tensor(x_d_window))
            batch_y.append(torch.tensor(y_window))
            batch_x_s.append(torch.tensor(x_s))

        batch_x_d = torch.stack(batch_x_d, dim=0)
        batch_y = torch.stack(batch_y, dim=0)
        batch_x_s = torch.stack(batch_x_s, dim=0)

        return {
            "x_d": batch_x_d,
            "x_s": batch_x_s,
            "y": batch_y,
        }
