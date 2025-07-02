import random
from pathlib import Path
from typing import Any, Union

import pandas as pd
from ruamel.yaml import YAML


class Config:
    """
    Lightweight config class for space-time LSTM training.
    
    Can read from:
    - YAML file
    - Python dictionary
    
    All keys ending in _dir, _file, or _path are converted to Path objects.
    All keys ending in _date are converted to pandas.Timestamp.
    """

    def __init__(self, yml_path_or_dict: Union[Path, dict]):
        if isinstance(yml_path_or_dict, Path):
            self._cfg = self._read_yaml(yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            self._cfg = yml_path_or_dict
        else:
            raise ValueError(f"Unsupported config input type: {type(yml_path_or_dict)}")

        self._parse_paths()
        self._parse_dates()

    def _read_yaml(self, path: Path) -> dict:
        yaml = YAML(typ="safe")
        with open(path, "r") as f:
            return yaml.load(f)

    def _parse_paths(self):
        """Convert all *_dir, *_file, *_path keys to Path objects."""
        for k, v in self._cfg.items():
            if any(k.endswith(x) for x in ["_dir", "_file", "_path"]):
                if isinstance(v, list):
                    self._cfg[k] = [Path(x) for x in v]
                elif v is not None:
                    self._cfg[k] = Path(v)

    def _parse_dates(self):
        """Convert all *_date keys to pandas.Timestamp."""
        for k, v in self._cfg.items():
            if k.endswith("_date"):
                if isinstance(v, list):
                    self._cfg[k] = [pd.to_datetime(x, format="%d/%m/%Y") for x in v]
                elif v is not None:
                    self._cfg[k] = pd.to_datetime(v, format="%d/%m/%Y")

    def as_dict(self) -> dict:
        """Return config as dictionary."""
        return self._cfg

    def dump(self, out_path: Path):
        yaml = YAML()
        yaml.default_flow_style = False
        # convert Paths back to strings for saving
        save_dict = {}
        for k, v in self._cfg.items():
            if isinstance(v, Path):
                save_dict[k] = str(v)
            elif isinstance(v, list) and all(isinstance(x, Path) for x in v):
                save_dict[k] = [str(x) for x in v]
            elif isinstance(v, pd.Timestamp):
                save_dict[k] = v.strftime("%d/%m/%Y")
            else:
                save_dict[k] = v
        with open(out_path, "w") as f:
            yaml.dump(save_dict, f)

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __getattr__(self, item: str) -> Any:
        # allows config.something syntax
        try:
            return self._cfg[item]
        except KeyError:
            raise AttributeError(f"No such config key: {item}")
