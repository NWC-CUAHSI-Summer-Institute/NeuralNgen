"""
bmi_CatchmentLSTM: BMI (Basic Model Interface, CSDMS BMI 2.0) wrapper around
CatchmentLSTM, modeled on the NOAA-OWP/Lynker reference bmi_lstm.py.

Runs CatchmentLSTM one hour at a time inside NextGen via
CatchmentLSTM.forward_step(), carrying (h, c) hidden state forward between
calls. The prediction only actually changes once every 24 calls (once a
full calendar day of hourly hidden states has been mean-pooled and passed
through the trained head, exactly reproducing forward()'s day-block
behavior) - the model has no hourly-resolution training signal to draw on,
so held-constant-for-23-hours is the faithful output, not a hack.

FIXED vs the previous draft of this file: that version called
`self._model(dyn_norm, static_norm, hidden_state=(h, c))` - i.e. forward()
itself with a hidden_state kwarg that doesn't exist, on a T=1 window that
forward() would also reject (t_hours must be a multiple of 24). That
call contract was never exercised by train_multi_gauge.py, which only ever
calls `model(x_d, x_s)` on full multi-day windows. This version calls
forward_step() instead, which is the actual stateful single-hour entry
point added to catchment_lstm.py, verified numerically identical to
forward() when called 24x in a row.
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
        # Training target is flow_m3s (m^3/s at the GAUGE) and training aggregates catchments as an
        # area-weighted mean, so catchment i contributes pred*A_i/A_gauge. ngen multiplies a depth by
        # A_i*1e6/3600, so the depth giving that contribution is pred*3600/(A_gauge*1e6).
        self.gauge_area_sqkm = float(cfg["gauge_area_sqkm"])
        self.output_factor_m = 3600.0 / (self.gauge_area_sqkm * 1.0e6)  # gauge m3/s -> m depth per 1 h

        self._time_step_size = float(cfg.get("time_step_size", 3600.0))
        self._current_time = 0.0
        self._start_time = 0.0

        # forward_step() initializes (h, c) itself from static_norm on the
        # first call when passed None - do NOT zero-init here, that would
        # skip the trained static-conditioning MLP (static_to_h0/c0) and
        # silently start every run from the wrong initial state.
        self.h = None
        self.c = None

        self._values = {name: 0.0 for name in self._input_var_names}
        self._values["land_surface_water__runoff_volume_flux"] = 0.0

        for name in self._input_var_names:
            self._var_units[name] = "unknown"
        self._var_units["land_surface_water__runoff_volume_flux"] = "m"

    # ------------------------------------------------------------------
    # BMI: update
    # ------------------------------------------------------------------
    def update(self):
        dyn_vals = np.array(
            [self._values[name] for name in self.dynamic_inputs], dtype=np.float32
        )
        dyn_t = torch.as_tensor(dyn_vals, dtype=torch.float32).view(1, -1)  # (1, dyn_size) - ONE hour, no T dim
        dyn_norm = (dyn_t - self.dyn_mean) / self.dyn_std

        with torch.no_grad():
            pred_norm, self.h, self.c = self._model.forward_step(
                dyn_norm, self.static_norm, self.h, self.c
            )

        pred_physical_cms = float(pred_norm.item()) * self.q_std + self.q_mean
        pred_physical_cms = max(pred_physical_cms, 0.0)  # streamflow is non-negative

        runoff_m_per_timestep = pred_physical_cms * self.output_factor_m
        self._values["land_surface_water__runoff_volume_flux"] = runoff_m_per_timestep

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

    def get_value_at_indices(self, name, dest, inds):
        dest[:] = self._values[name]
        return dest

    def set_value(self, name, values):
        self._values[name] = float(np.asarray(values).reshape(-1)[0])

    def set_value_at_indices(self, name, inds, src):
        self._values[name] = float(np.asarray(src).reshape(-1)[0])

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

    def get_grid_shape(self, grid, shape):
        raise NotImplementedError("scalar grid (grid 0) has no shape - not a structured grid")

    def get_grid_spacing(self, grid, spacing):
        raise NotImplementedError("scalar grid (grid 0) has no spacing - not a structured grid")

    def get_grid_origin(self, grid, origin):
        raise NotImplementedError("scalar grid (grid 0) has no origin - not a structured grid")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("scalar grid (grid 0) has no coordinates - not a mesh")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("scalar grid (grid 0) has no coordinates - not a mesh")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("scalar grid (grid 0) has no coordinates - not a mesh")

    def get_grid_node_count(self, grid):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("scalar grid (grid 0) is not an unstructured mesh")

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