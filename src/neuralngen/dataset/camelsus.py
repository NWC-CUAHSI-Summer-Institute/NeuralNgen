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
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str = None,
        additional_features: List[Dict[str, pd.DataFrame]] = [],
        id_to_int: Dict[str, int] = {},
        scaler: Dict[str, Union[pd.Series, float]] = {},
    ):
        super().__init__(
            cfg=cfg,
            period=period
        )

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement hourly basin loading.")

    def _load_static_attributes(self) -> pd.DataFrame:
        return load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """
    Load static attributes from CAMELS attribute files.
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

    # HUC codes
    if 'huc_02' in df.columns:
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError("Some basins are missing static attributes.")
        df = df.loc[basins]

    return df