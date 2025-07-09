# src/neuralngen/dataset/basedataset.py

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset
from neuralngen.dataset.normalization import (
    compute_scalers,
    apply_scalers,
    save_scalers,
    load_scalers,
)

class BaseDataset(Dataset):
    def __init__(self, cfg, period, run_dir=None, do_load_scalers=True):
        """
        Base class for datasets with dynamic and static inputs.

        Parameters
        ----------
        cfg : Config
            Configuration object.
        period : str
            One of {"train", "validation", "test"}
        run_dir : str or Path, optional
            Directory for saving/loading scalers.
        do_load_scalers : bool
            Whether to compute/load and apply scalers.
        """
        self.cfg = cfg
        self.period = period
        self.all_basins_with_samples = []

        # -----------------------------------------
        # Load list of basin IDs for this period
        # -----------------------------------------
        basin_file = getattr(cfg, f"{period}_basin_file")
        with open(basin_file, 'r') as f:
            self.basins = [line.strip() for line in f.readlines()]

        # -----------------------------------------
        # Load static attributes for all basins
        # -----------------------------------------
        static_attr_df = self._load_static_attributes()
        self.static_attributes = {}
        self.basin_info = {}

        # Collect static dataframes for scaling
        static_dfs = []
        for basin in self.basins:
            attr_names = cfg.static_attributes
            for attr in attr_names:
                if attr not in static_attr_df.columns:
                    raise ValueError(f"Static attribute {attr} missing from attribute file.")
            static_dfs.append(static_attr_df.loc[basin, attr_names].to_frame().T)

        # Compute or load static scalers
        static_scaler_file = None
        if do_load_scalers:
            if run_dir is None:
                raise ValueError("Cannot compute/load scalers without run_dir.")
            scalers_dir = Path(run_dir) / "scalers"
            scalers_dir.mkdir(parents=True, exist_ok=True)
            static_scaler_file = scalers_dir / "static.json"
            if period == "train":
                self.static_scalers = compute_scalers(static_dfs, cfg.static_attributes)
                save_scalers(self.static_scalers, static_scaler_file)
                print("[INFO] Computed and saved static scalers for training data.")
            else:
                self.static_scalers = load_scalers(static_scaler_file)
                print(f"[INFO] Loaded static scalers from disk for period: {period}")
        else:
            self.static_scalers = None

        # Apply static scaling and store
        for basin in self.basins:
            attr_names = cfg.static_attributes
            raw_vals = static_attr_df.loc[basin, attr_names]
            if self.static_scalers is not None:
                df_tmp = raw_vals.to_frame().T
                scaled_df = apply_scalers(df_tmp, self.static_scalers)
                numeric_values = scaled_df.values.astype(np.float32).reshape(-1)
            else:
                numeric_values = raw_vals.values.astype(np.float32)
            self.static_attributes[basin] = numeric_values

            # Basin info
            lat = static_attr_df.loc[basin, 'gauge_lat'] if 'gauge_lat' in static_attr_df.columns else np.nan
            lon = static_attr_df.loc[basin, 'gauge_lon'] if 'gauge_lon' in static_attr_df.columns else np.nan
            area_km2 = static_attr_df.loc[basin, 'area_gages2'] if 'area_gages2' in static_attr_df.columns else np.nan
            self.basin_info[basin] = {
                "gauge_id": basin,
                "gauge_lat": float(lat),
                "gauge_lon": float(lon),
                "basin_area_km2": float(area_km2),
            }

        # ------------------------------------------------------
        # Load dynamic inputs and targets for each basin
        # ------------------------------------------------------
        dyn_dfs = []
        target_dfs = []
        self.dynamic_inputs = {}
        self.targets = {}
        self.targets_raw = {}

        for basin in self.basins:
            df = self._load_basin_data(basin)
            start_date = pd.to_datetime(getattr(cfg, f"{period}_start_date"))
            end_date = pd.to_datetime(getattr(cfg, f"{period}_end_date"))
            df = df.loc[start_date:end_date]
            df = df.dropna(subset=cfg.dynamic_inputs + cfg.target_variables)

            dyn_dfs.append(df[cfg.dynamic_inputs])
            target_dfs.append(df[cfg.target_variables])
            self.dynamic_inputs[basin] = df[cfg.dynamic_inputs]
            self.targets[basin] = df[cfg.target_variables]
            self.targets_raw[basin] = df[cfg.target_variables].copy()

        # ------------------------------------------------------
        # Compute or load dynamic & target scalers, then apply
        # ------------------------------------------------------
        if do_load_scalers:
            # assume scalers_dir exists
            dyn_scaler_file = Path(run_dir) / "scalers" / "dynamic_inputs.json"
            target_scaler_file = Path(run_dir) / "scalers" / "targets.json"
            if period == "train":
                self.dyn_scalers = compute_scalers(dyn_dfs, cfg.dynamic_inputs)
                self.target_scalers = compute_scalers(target_dfs, cfg.target_variables)
                save_scalers(self.dyn_scalers, dyn_scaler_file)
                save_scalers(self.target_scalers, target_scaler_file)
                print("[INFO] Computed and saved dynamic & target scalers for training data.")
            else:
                self.dyn_scalers = load_scalers(dyn_scaler_file)
                self.target_scalers = load_scalers(target_scaler_file)
                print(f"[INFO] Loaded dynamic & target scalers for period: {period}")

            for basin in self.basins:
                # dynamic
                dyn_df = pd.DataFrame(self.dynamic_inputs[basin], columns=cfg.dynamic_inputs)
                dyn_scaled = apply_scalers(dyn_df, self.dyn_scalers)
                self.dynamic_inputs[basin] = dyn_scaled.values.astype(np.float32)
                # target
                tgt_df = pd.DataFrame(self.targets[basin], columns=cfg.target_variables)
                tgt_scaled = apply_scalers(tgt_df, self.target_scalers)
                self.targets[basin] = tgt_scaled.values.astype(np.float32)
        else:
            for basin in self.basins:
                self.dynamic_inputs[basin] = self.dynamic_inputs[basin].values.astype(np.float32)
                self.targets[basin] = self.targets[basin].values.astype(np.float32)

        # ------------------------------------------------------
        # Build sliding windows from all basins
        # ------------------------------------------------------
        self.samples = []
        self.samples_by_basin = {basin: [] for basin in self.basins}
        self.window_size = cfg.batch_timesteps + cfg.sequence_length

        for basin in self.basins:
            arr = self.dynamic_inputs[basin]
            if arr.shape[0] < self.window_size:
                continue
            n_windows = arr.shape[0] - self.window_size + 1
            for start_idx in range(n_windows):
                self.samples.append((basin, start_idx))
                self.samples_by_basin[basin].append(start_idx)

        for basin, starts in self.samples_by_basin.items():
            if starts:
                self.all_basins_with_samples.append(basin)

        if len(self.all_basins_with_samples) == 0:
            print("[WARNING] No basins contain enough data for any samples!")
        print(f"Built dataset. Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basin, start_idx = self.samples[idx]
        return self._load_window(basin, start_idx)

    def sample_window_for_basin(self, basin):
        start_idx = random.choice(self.samples_by_basin[basin])
        return self._load_window(basin, start_idx)

    def _load_window(self, basin, start_idx):
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

    @property
    def total_windows(self):
        return sum(len(w) for w in self.samples_by_basin.values())

# import torch
# import numpy as np
# import pandas as pd
# import random
# from pathlib import Path
# from torch.utils.data import Dataset
# from neuralngen.dataset.normalization import (
#     compute_scalers,
#     apply_scalers,
#     save_scalers,
#     load_scalers,
# )

# class BaseDataset(Dataset):
#     def __init__(self, cfg, period, run_dir=None, do_load_scalers=True):
#         """
#         Base class for datasets with dynamic and static inputs.

#         Parameters
#         ----------
#         cfg : Config
#             Configuration object.
#         period : str
#             One of {"train", "validation", "test"}
#         run_dir : str or Path, optional
#             Directory for saving/loading scalers.
#         do_load_scalers : bool
#             Whether to compute/load and apply scalers.
#         """
#         self.cfg = cfg
#         self.period = period
#         self.all_basins_with_samples = []

#         # -----------------------------------------
#         # Load list of basin IDs for this period
#         # -----------------------------------------
#         basin_file = getattr(cfg, f"{period}_basin_file")
#         with open(basin_file, 'r') as f:
#             self.basins = [line.strip() for line in f.readlines()]

#         # -----------------------------------------
#         # Load static attributes for all basins
#         # -----------------------------------------
#         static_attr_df = self._load_static_attributes()

#         self.static_attributes = {}
#         self.basin_info = {}

#         for basin in self.basins:
#             attr_names = self.cfg.static_attributes

#             # Validate all required attributes exist
#             for attr in attr_names:
#                 if attr not in static_attr_df.columns:
#                     raise ValueError(f"Static attribute {attr} missing from attribute file.")

#             numeric_values = static_attr_df.loc[basin, attr_names].values.astype(np.float32)
#             self.static_attributes[basin] = numeric_values

#             lat = static_attr_df.loc[basin, 'gauge_lat'] if 'gauge_lat' in static_attr_df.columns else np.nan
#             lon = static_attr_df.loc[basin, 'gauge_lon'] if 'gauge_lon' in static_attr_df.columns else np.nan
#             area_km2 = static_attr_df.loc[basin, 'area_gages2'] if 'area_gages2' in static_attr_df.columns else np.nan

#             self.basin_info[basin] = {
#                 "gauge_id": basin,
#                 "gauge_lat": float(lat),
#                 "gauge_lon": float(lon),
#                 "basin_area_km2": float(area_km2),
#             }

#         # ------------------------------------------------------
#         # Load dynamic inputs and targets for each basin
#         # ------------------------------------------------------
#         dyn_dfs = []
#         target_dfs = []

#         self.dynamic_inputs = {}
#         self.targets = {}

#         self.targets_raw = {}

#         for basin in self.basins:
#             df = self._load_basin_data(basin)

#             start_date = pd.to_datetime(getattr(cfg, f"{period}_start_date"))
#             end_date = pd.to_datetime(getattr(cfg, f"{period}_end_date"))

#             # Restrict to desired period
#             df = df.loc[start_date:end_date]

#             # Drop rows missing any dynamic inputs or targets
#             df = df.dropna(subset=cfg.dynamic_inputs + cfg.target_variables)

#             # Convert QObs columns from m³/s → mm/hr if needed
#             area_km2 = static_attr_df.loc[basin, 'area_gages2']

#             dyn_dfs.append(df[cfg.dynamic_inputs])
#             target_dfs.append(df[cfg.target_variables])

#             self.dynamic_inputs[basin] = df[cfg.dynamic_inputs]
#             self.targets[basin] = df[cfg.target_variables]
#             # PATCH: Store original (unscaled) target DataFrame
#             self.targets_raw[basin] = df[cfg.target_variables].copy()

#         # ------------------------------------------------------
#         # Compute or load scalers, and apply them to all basins
#         # ------------------------------------------------------
#         if do_load_scalers:
#             if run_dir is None:
#                 raise ValueError("Cannot compute/load scalers without run_dir.")

#             scalers_dir = Path(run_dir) / "scalers"
#             scalers_dir.mkdir(parents=True, exist_ok=True)

#             dyn_scaler_file = scalers_dir / "dynamic_inputs.json"
#             target_scaler_file = scalers_dir / "targets.json"

#             if period == "train":
#                 # Compute new scalers on training data
#                 dyn_scalers = compute_scalers(dyn_dfs, cfg.dynamic_inputs)
#                 target_scalers = compute_scalers(target_dfs, cfg.target_variables)

#                 save_scalers(dyn_scalers, dyn_scaler_file)
#                 save_scalers(target_scalers, target_scaler_file)

#                 self.dyn_scalers = dyn_scalers
#                 self.target_scalers = target_scalers

#                 print("[INFO] Computed and saved scalers for training data.")
#             else:
#                 # Load scalers from disk (must already exist from training)
#                 self.dyn_scalers = load_scalers(dyn_scaler_file)
#                 self.target_scalers = load_scalers(target_scaler_file)

#                 print(f"[INFO] Loaded scalers from disk for period: {period}")

#             # Apply scaling
#             for basin in self.basins:
#                 dyn_df_scaled = apply_scalers(self.dynamic_inputs[basin], self.dyn_scalers)
#                 target_df_scaled = apply_scalers(self.targets[basin], self.target_scalers)
#                 self.dynamic_inputs[basin] = dyn_df_scaled.values.astype(np.float32)
#                 self.targets[basin] = target_df_scaled.values.astype(np.float32)

#         else:
#             # No scaling at all
#             for basin in self.basins:
#                 self.dynamic_inputs[basin] = self.dynamic_inputs[basin].values.astype(np.float32)
#                 self.targets[basin] = self.targets[basin].values.astype(np.float32)

#         # ------------------------------------------------------
#         # Build sliding windows from all basins
#         # ------------------------------------------------------
#         self.samples = []
#         self.samples_by_basin = {basin: [] for basin in self.basins}
#         self.window_size = cfg.batch_timesteps + cfg.sequence_length

#         for basin in self.basins:
#             dyn = self.dynamic_inputs[basin]
#             if dyn.shape[0] < self.window_size:
#                 continue

#             n_windows = dyn.shape[0] - self.window_size + 1
#             for start_idx in range(n_windows):
#                 self.samples.append((basin, start_idx))
#                 self.samples_by_basin[basin].append(start_idx)

#         # Identify basins that actually have samples
#         for basin, starts in self.samples_by_basin.items():
#             if len(starts) > 0:
#                 self.all_basins_with_samples.append(basin)

#         if len(self.all_basins_with_samples) == 0:
#             print("[WARNING] No basins contain enough data for any samples!")

#         print(f"Built dataset. Number of samples: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         """
#         Fetch a single training sample window by global index.
#         """
#         basin, start_idx = self.samples[idx]
#         return self._load_window(basin, start_idx)

#     def sample_window_for_basin(self, basin):
#         """
#         Randomly select one window for a specific basin.
#         """
#         start_idx = random.choice(self.samples_by_basin[basin])
#         return self._load_window(basin, start_idx)

#     def _load_window(self, basin, start_idx):
#         """
#         Slice a window of length window_size for a single basin.

#         Returns a dict with:
#           - x_d: dynamic inputs [window_size, Din]
#           - y: targets [window_size, Dout]
#           - x_s: static attributes [S]
#           - x_info: metadata dict
#         """
#         end_idx = start_idx + self.window_size

#         x_d_window = self.dynamic_inputs[basin][start_idx:end_idx, :]
#         y_window = self.targets[basin][start_idx:end_idx, :]

#         x_s = self.static_attributes[basin]
#         x_info = self.basin_info[basin]

#         return {
#             "x_d": torch.tensor(x_d_window, dtype=torch.float32),
#             "x_s": torch.tensor(x_s, dtype=torch.float32),
#             "y": torch.tensor(y_window, dtype=torch.float32),
#             "x_info": x_info,
#         }

#     def _load_static_attributes(self):
#         """
#         Must be implemented by subclasses.
#         """
#         raise NotImplementedError()

#     def _load_basin_data(self, basin):
#         """
#         Must be implemented by subclasses.
#         """
#         raise NotImplementedError()

#     @property
#     def total_windows(self):
#         """
#         Return the total number of time windows across all basins.
#         """
#         return sum(len(windows) for windows in self.samples_by_basin.values())