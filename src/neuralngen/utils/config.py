# src/neuralngen/utils/config.py

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
    Nested dictionaries are automatically converted to Config objects.
    """

    def __init__(self, yml_path_or_dict: Union[Path, dict]):
        if isinstance(yml_path_or_dict, Path):
            raw = self._read_yaml(yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            raw = yml_path_or_dict
        else:
            raise ValueError(f"Unsupported config input type: {type(yml_path_or_dict)}")

        self._cfg = self._wrap_nested(raw)
        self._parse_paths()
        self._parse_dates()

    def _read_yaml(self, path: Path) -> dict:
        yaml = YAML(typ="safe")
        with open(path, "r") as f:
            return yaml.load(f)

    def _wrap_nested(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._wrap_nested(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._wrap_nested(v) for v in data]
        else:
            return data

    def _parse_paths(self):
        for k, v in self._cfg.items():
            if any(k.endswith(x) for x in ["_dir", "_file", "_path"]):
                if isinstance(v, list):
                    self._cfg[k] = [Path(x) if not isinstance(x, Path) else x for x in v]
                elif v is not None and not isinstance(v, Path):
                    self._cfg[k] = Path(v)

    def _parse_dates(self):
        for k, v in self._cfg.items():
            if k.endswith("_date"):
                if isinstance(v, list):
                    self._cfg[k] = [pd.to_datetime(x, format="%d/%m/%Y") for x in v]
                elif v is not None:
                    self._cfg[k] = pd.to_datetime(v, format="%d/%m/%Y")

    def as_dict(self) -> dict:
        return self._unwrap(self._cfg)

    def _unwrap(self, obj: Any) -> Any:
        """Convert nested Config objects back to dictionaries for serialization."""
        if isinstance(obj, Config):
            return obj.as_dict()
        elif isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._unwrap(x) for x in obj]
        else:
            return obj

    def dump(self, out_path: Path):
        yaml = YAML()
        yaml.default_flow_style = False

        save_dict = self.as_dict()

        # Convert special types
        for k, v in save_dict.items():
            if isinstance(v, Path):
                save_dict[k] = str(v)
            elif isinstance(v, list) and all(isinstance(x, Path) for x in v):
                save_dict[k] = [str(x) for x in v]
            elif isinstance(v, pd.Timestamp):
                save_dict[k] = v.strftime("%d/%m/%Y")

        with open(out_path, "w") as f:
            yaml.dump(save_dict, f)

    def __getitem__(self, key: str) -> Any:
        value = self._cfg[key]
        if isinstance(value, dict):
            return Config(value)
        else:
            return value

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._cfg[item]
            if isinstance(value, dict):
                return Config(value)
            else:
                return value
        except KeyError:
            raise AttributeError(f"No such config key: {item}")

    def __repr__(self):
        return f"Config({self._cfg})"