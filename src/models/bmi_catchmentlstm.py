"""
bmi_CatchmentLSTM: BMI (Basic Model Interface, CSDMS BMI 2.0) wrapper around
CatchmentLSTM, modeled on the NOAA-OWP/Lynker reference bmi_lstm.py.

Runs CatchmentLSTM one timestep at a time inside NextGen: each update()
call feeds the current forcing values through the LSTM as a length-1
sequence, carrying (h, c) hidden state forward between calls so the model
stays stateful across the full simulation without needing the whole
forcing series in memory at once.
"""

import time
import yaml
import numpy as np
import torch

from bmipy import Bmi
from src.models.catchment_lstm import CatchmentLSTM


class bmi_CatchmentLSTM(Bmi):

    _name = "CatchmentLSTM"
    _input_var_names = []   # populated in initialize() from config dynamic_inputs
    _output_var_names = ["land_surface_water__runoff_volume_flux"]

    def __init__(self):
        self._model = None
        self._values = {}
        self._var_units = {}
        self._start_time = 0.0
        self._current_time = 0.0
        self._end_time = np.finfo("d").max
        self._time_units = "s"
        self._time_step_size = 3600.0  # hourly

        self.h = None
        self.c = None

    # ------------------------------------------------------------------
    # BMI: initialize
    # ------------------------------------------------------------------
    def initialize(self, bmi_cfg_file=None):
        with open(bmi_cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg_bmi = cfg

        ckpt = torch.load(cfg["train_cfg_file"], map_location="cpu", weights_only=False)
        train_cfg = ckpt["config"]

        self.dynamic_inputs = train_cfg["dynamic_inputs"]
        self.static_attributes = train_cfg["static_attributes"]
        self._input_var_names = list(self.dynamic_inputs)

        self.dyn_mean = torch.as_tensor(ckpt["dyn_mean"], dtype=torch.float32)
        self.dyn_std = torch.as_tensor(ckpt["dyn_std"], dtype=torch.float32)
        self.stat_mean = torch.as_tensor(ckpt["stat_mean"], dtype=torch.float32)
        self.stat_std = torch.as_tensor(ckpt["stat_std"], dtype=torch.float32)

        gauge_id = cfg["gauge_id"]
        self.q_mean, self.q_std = ckpt["q_stats"][gauge_id]

        self._model = CatchmentLSTM(
            dynamic_size=len(self.dynamic_inputs),
            static_size=len(self.static_attributes),
            hidden_size=train_cfg.get("hidden_size", 64),
            num_layers=train_cfg.get("num_layers", 2),
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        static_vals = np.array(
            [cfg["static_attributes"][name] for name in self.static_attributes],
            dtype=np.float32,
        )
        static_t = torch.as_tensor(static_vals, dtype=torch.float32).unsqueeze(0)  # (1, static_size)
        self.static_norm = (static_t - self.stat_mean) / self.stat_std

        self.area_sqkm = float(cfg["area_sqkm"])
        # mm/h -> m^3/s : (1/1000 m/mm) * (area_sqkm * 1e6 m^2/km^2) * (1/3600 h/s)
        self.output_factor_cms = (1.0 / 1000.0) * (self.area_sqkm * 1_000_000.0) * (1.0 / 3600.0)

        self._time_step_size = float(cfg.get("time_step_size", 3600.0))
        self._current_time = 0.0
        self._start_time = 0.0

        self.h = torch.zeros(self._model.num_layers, 1, self._model.hidden_size)
        self.c = torch.zeros(self._model.num_layers, 1, self._model.hidden_size)

        self._values = {name: 0.0 for name in self._input_var_names}
        self._values["land_surface_water__runoff_volume_flux"] = 0.0

        for name in self._input_var_names:
            self._var_units[name] = "unknown"
        self._var_units["land_surface_water__runoff_volume_flux"] = "m3 s-1"

    # ------------------------------------------------------------------
    # BMI: update
    # ------------------------------------------------------------------
    def update(self):
        dyn_vals = np.array(
            [self._values[name] for name in self.dynamic_inputs], dtype=np.float32
        )
        dyn_t = torch.as_tensor(dyn_vals, dtype=torch.float32).view(1, 1, -1)  # (1, T=1, dyn_size)
        dyn_norm = (dyn_t - self.dyn_mean) / self.dyn_std

        with torch.no_grad():
            pred_norm, (self.h, self.c) = self._model(
                dyn_norm, self.static_norm, hidden_state=(self.h, self.c)
            )

        pred_physical_mm_h = float(pred_norm.item()) * self.q_std + self.q_mean
        pred_physical_mm_h = max(pred_physical_mm_h, 0.0)  # streamflow is non-negative

        runoff_cms = pred_physical_mm_h * self.output_factor_cms
        self._values["land_surface_water__runoff_volume_flux"] = runoff_cms

        self._current_time += self._time_step_size

    def update_until(self, time_):
        while self._current_time < time_:
            self.update()

    def finalize(self):
        self._model = None

    # ------------------------------------------------------------------
    # BMI: variable getters/setters
    # ------------------------------------------------------------------
    def get_value(self, name, dest):
        dest[:] = self._values[name]
        return dest

    def get_value_ptr(self, name):
        return np.array([self._values[name]])

    def set_value(self, name, values):
        self._values[name] = float(np.asarray(values).reshape(-1)[0])

    def get_var_type(self, name):
        return "float64"

    def get_var_units(self, name):
        return self._var_units.get(name, "unknown")

    def get_var_itemsize(self, name):
        return 8

    def get_var_nbytes(self, name):
        return 8

    def get_var_location(self, name):
        return "node"

    def get_var_grid(self, name):
        return 0

    def get_grid_type(self, grid):
        return "scalar"

    def get_grid_rank(self, grid):
        return 1

    def get_grid_size(self, grid):
        return 1

    def get_input_var_names(self):
        return tuple(self._input_var_names)

    def get_output_var_names(self):
        return tuple(self._output_var_names)

    def get_component_name(self):
        return self._name

    def get_input_item_count(self):
        return len(self._input_var_names)

    def get_output_item_count(self):
        return len(self._output_var_names)

    # ------------------------------------------------------------------
    # BMI: time
    # ------------------------------------------------------------------
    def get_start_time(self):
        return self._start_time

    def get_current_time(self):
        return self._current_time

    def get_end_time(self):
        return self._end_time

    def get_time_step(self):
        return self._time_step_size

    def get_time_units(self):
        return self._time_units
