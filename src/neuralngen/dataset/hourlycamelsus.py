# src/neuralngen/dataset/hourlycamels.py

from pathlib import Path
import pandas as pd
import numpy as np

from neuralngen.dataset.camelsus import CamelsUS

class HourlyCamelsDataset(CamelsUS):
    """
    Dataset for hourly CAMELS US data stored as CSVs.

    Supports either AORC or NLDAS atmospheric forcings.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    is_train : bool
        Whether in training mode.
    period : str
        One of {"train", "validation", "test"}
    """

    def __init__(
        self,
        cfg,
        is_train: bool,
        period: str,
        basin: str = None,
        additional_features=[],
        id_to_int={},
        scaler={},
        run_dir=None,
        do_load_scalers=True,
    ):
        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features,
            id_to_int=id_to_int,
            scaler=scaler,
            run_dir=run_dir,
            do_load_scalers=do_load_scalers,
        )

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """
        Load hourly forcing and discharge data for a basin from either AORC or NLDAS.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with merged atmospheric forcings and discharge.
        """

        # -------------------------------
        # 1. Load hourly forcing data
        # -------------------------------
        hourly_dir = Path(self.cfg.data_dir) / "hourly" / self.cfg.forcings
        matching_files = list(hourly_dir.glob(f"{basin}_*.csv"))
        if not matching_files:
            raise FileNotFoundError(
                f"No hourly CSV found for basin {basin} in {hourly_dir}."
            )

        fpath = matching_files[0]
        forcing_df = pd.read_csv(fpath, index_col=0, parse_dates=True)

        # -------------------------------
        # 2. Load hourly discharge
        # -------------------------------
        discharge_df = load_hourly_us_discharge(
            Path(self.cfg.data_dir), basin
        )

        # Merge discharge into forcings
        df = forcing_df.join(discharge_df, how="left")

        # Replace negative QObs values with NaN
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            num_neg = (df[col] < -1).sum()
            if num_neg > 0:
                print(f"[WARNING] {num_neg} negative values in {col} before replacement.")
            df.loc[df[col] < -1, col] = np.nan

        # Check again after replacement
        for col in qobs_cols:
            if (df[col] < -1).any():
                print(f"[ERROR] Negative values remain in {col} even after replacement.")

        # Check for missing required columns
        required_cols = self.cfg.dynamic_inputs + self.cfg.target_variables
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Basin {basin} missing columns: {missing_cols}"
            )

        return df
    

def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    pattern = '**/*usgs-hourly.csv'
    discharge_path = data_dir / 'hourly' / 'usgs_streamflow'
    files = list(discharge_path.glob(pattern))

    if len(files) == 0:
        discharge_path = discharge_path.parent / 'usgs-streamflow'
        files = list(discharge_path.glob(pattern))

    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(
            f'No discharge file for Basin {basin} at {discharge_path}'
        )

    df = pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])
    df = df[["QObs(mm/h)"]]
    return df