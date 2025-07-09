# src/neuralngen/dataset/camelsus.py

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from neuralngen.dataset.basedataset import BaseDataset
from neuralngen.utils import Config

class CamelsUS(BaseDataset):
    """
    Base class for CAMELS US datasets, no daily data handling anymore.

    All specific implementations should override `_load_basin_data`.
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
        self.is_train = is_train
        self.basin = basin
        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler

        super().__init__(
            cfg=cfg,
            period=period,
            run_dir=run_dir,
            do_load_scalers=do_load_scalers,
        )

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement hourly basin loading.")

    def _load_static_attributes(self) -> pd.DataFrame:
        return load_camels_us_attributes(self.cfg.data_dir, 
                                         basins=self.basins, 
                                         attributes_to_keep=self.cfg.static_attributes)


def load_camels_us_attributes(
    data_dir: Path,
    basins: List[str] = [],
    attributes_to_keep: List[str] = []
) -> pd.DataFrame:
    """
    Load static attributes from CAMELS attribute files,
    keeping only selected attributes plus lat/lon.

    Parameters
    ----------
    data_dir : Path
        Root CAMELS data directory.
    basins : List[str]
        List of basin IDs to load.
    attributes_to_keep : List[str]
        Names of attributes to keep as static features.
        lat, lon are always kept separately.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by gauge_id, containing only
        attributes_to_keep plus lat/lon.
    """
    attributes_path = Path(data_dir) / 'camels_attributes_v2.0'
    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    # Add derived HUC code if present
    if 'huc_02' in df.columns:
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError("Some basins are missing static attributes.")
        df = df.loc[basins]

    # Always keep lat/lon separately
    required_cols = ['gauge_lat', 'gauge_lon', 'area_gages2']
    cols_to_keep = required_cols.copy()

    # Only keep attributes listed in config
    if attributes_to_keep:
        for attr in attributes_to_keep:
            if attr in df.columns:
                cols_to_keep.append(attr)
            else:
                raise ValueError(f"Attribute {attr} not found in CAMELS attributes.")

    # Drop any columns not explicitly selected
    df = df.loc[:, cols_to_keep]

    return df