# src/neuralngen/dataset/dailycamelsus.py

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from neuralngen.dataset.camelsus import CamelsUS


class DailyCamelsDataset(CamelsUS):
    """
    Dataset for daily CAMELS US data.
    Loads daily forcings and discharge using CAMELS file structure.
    """

    def __init__(
        self,
        cfg,
        is_train: bool,
        period: str,
        basin: str = None,
        additional_features: list = [],
        id_to_int: dict = {},
        scaler: dict = {},
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
        Load daily forcing and discharge data for a basin.

        Parameters
        ----------
        basin : str
            8-digit USGS basin identifier.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with forcing columns and QObs(mm/d) target.
        """
        # 1. Forcing data
        forcing = self.cfg.forcings  # e.g., 'nldas', 'daymet'
        df_forcing, area = load_camels_us_forcings(
            self.cfg.data_dir, basin, forcing
        )

        df = df_forcing.copy()

        # 2. Discharge
        df['QObs(mm/d)'] = load_camels_us_discharge(
            self.cfg.data_dir, basin, area
        )

        # 3. Clean negative discharge
        qobs_cols = [c for c in df.columns if 'qobs' in c.lower()]
        df[qobs_cols] = df[qobs_cols].where(df[qobs_cols] >= 0, np.nan)

        # 4. Validate required columns
        required = self.cfg.dynamic_inputs + self.cfg.target_variables
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Basin {basin} missing columns: {missing}")

        return df


# -----------------------------------------------------------------------------
# Helper functions copied from NeuralHydrology
# -----------------------------------------------------------------------------

def load_camels_us_forcings(
    data_dir: Path,
    basin: str,
    forcings: str
) -> Tuple[pd.DataFrame, int]:
    """
    Load the daily CAMELS US forcing data for a basin.

    Parameters
    ----------
    data_dir : Path
        Root CAMELS data directory (must contain 'basin_mean_forcing').
    basin : str
        8-digit USGS basin identifier.
    forcings : str
        Name of forcing folder (e.g., 'nldas').

    Returns
    -------
    df : pd.DataFrame
        Time-indexed DataFrame of forcing variables.
    area : int
        Catchment area (m2) from the header.
    """
    forcing_path = Path(data_dir) / 'basin_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob(f'**/{basin}_*_forcing_leap.txt'))
    if not files:
        raise FileNotFoundError(f"No file for Basin {basin} at {files}")
    file_path = files[0]

    with open(file_path, 'r') as fp:
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        df = pd.read_csv(fp, sep='\s+')

    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
        format="%Y/%m/%d"
    )
    df = df.set_index("date")
    return df, area


def load_camels_us_discharge(
    data_dir: Path,
    basin: str,
    area: int
) -> pd.Series:
    """
    Load the daily CAMELS US discharge data and normalize to mm/day.

    Parameters
    ----------
    data_dir : Path
        Root CAMELS data directory (must contain 'usgs_streamflow').
    basin : str
        8-digit USGS basin identifier.
    area : int
        Catchment area (m2) from the forcing header.

    Returns
    -------
    pd.Series
        Time-indexed discharge in mm/day.
    """

    discharge_path = Path(data_dir) / 'usgs_streamflow'
    if not discharge_path.is_dir():
        raise OSError(f"{discharge_path} does not exist")

    files = list(discharge_path.glob(f'**/{basin}_streamflow_qc.txt'))
    if not files:
        raise FileNotFoundError(f"No file for Basin {basin} at {files}")
    file_path = files[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
        format="%Y/%m/%d"
    )
    df = df.set_index("date")

    # convert cfs to mm/day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)
    return df.QObs
