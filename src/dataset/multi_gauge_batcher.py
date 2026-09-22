"""
Per-gauge data handling and multi-gauge batch construction for training.

GaugeHandle       : lazy, per-gauge access to sub-catchment forcings
                     (NetCDF), static physiographic attributes and
                     catchment areas (from the hydrofabric geopackage), and
                     gauge-aggregate USGS streamflow observations.
MultiGaugeBatcher : cluster-proportional gauge selection across training
                     batches, and batch construction from GaugeHandles.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class GaugeHandle:
    """Lazy per-gauge handle to forcings, static attributes, areas, and
    observed streamflow for one CAMELS gauge."""

    def __init__(self, gauge_id: str, gauges_root, streamflow_dir,
                 dynamic_inputs, static_attributes):
        self.gauge_id = str(gauge_id)
        self.dynamic_inputs = list(dynamic_inputs)
        self.static_attributes = list(static_attributes)

        gauge_dir = Path(gauges_root) / f"gage-{self.gauge_id}"
        forcings_path = gauge_dir / "forcings" / "forcings.nc"
        gpkg_matches = list((gauge_dir / "config").glob("*_subset.gpkg"))
        if not gpkg_matches:
            raise FileNotFoundError(f"No subset gpkg found for gauge {self.gauge_id} under {gauge_dir}/config")
        gpkg_path = gpkg_matches[0]

        ds = xr.open_dataset(forcings_path)
        if "catchment-id" in ds.coords:
            self.catchment_ids = [str(c) for c in ds["catchment-id"].values]
        else:
            self.catchment_ids = [str(c) for c in ds["divide_id"].values]
        self.time_index = pd.to_datetime(ds["time"].values)
        self.n_catchments = len(self.catchment_ids)

        missing = [v for v in self.dynamic_inputs if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Gauge {self.gauge_id}: forcings.nc missing dynamic inputs {missing}")

        self._dyn = np.stack(
            [ds[var].values for var in self.dynamic_inputs], axis=-1
        ).astype(np.float32)  # (n_catchments, n_time, n_dynamic)
        ds.close()

        conn = sqlite3.connect(str(gpkg_path))
        # divide_id (e.g. "cat-2454") is the catchment-id key used throughout
        # this pipeline -- NOT the `id` column, which holds flowpath-style
        # "wb-XXXX" identifiers. Using `id` here silently produces garbage
        # area weights (confirmed root-cause bug from the first build).
        area_df = pd.read_sql(
            "SELECT divide_id, areasqkm FROM divides", conn
        ).set_index("divide_id")

        attr_cols = ", ".join(["divide_id"] + self.static_attributes)
        static_df = pd.read_sql(
            f"SELECT {attr_cols} FROM 'divide-attributes'", conn
        ).set_index("divide_id")
        conn.close()

        self.areas = area_df.reindex(self.catchment_ids)["areasqkm"].values.astype(np.float32)
        self.static = static_df.reindex(self.catchment_ids)[self.static_attributes].values.astype(np.float32)

        sf_path = Path(streamflow_dir) / f"{self.gauge_id}-usgs-hourly.csv"
        sf = pd.read_csv(sf_path, parse_dates=["date"]).set_index("date")
        self.q_obs = sf["QObs(mm/h)"].reindex(self.time_index)

    def get_window(self, start_idx: int, end_idx: int):
        """(dynamic, obs) for time indices [start_idx, end_idx).
        dynamic shape: (n_catchments, T, n_dynamic)."""
        dyn = self._dyn[:, start_idx:end_idx, :]
        obs = self.q_obs.values[start_idx:end_idx]
        return dyn, obs


class MultiGaugeBatcher:
    """Cluster-proportional gauge selection and batch construction across
    many GaugeHandles for multi-gauge training."""

    def __init__(self, gauge_handles: dict, clusters_csv, exclude_singleton_clusters=True):
        self.handles = gauge_handles  # {gauge_id: GaugeHandle}
        clusters = pd.read_csv(clusters_csv, dtype={"gauge_id": str}).set_index("gauge_id")
        clusters = clusters.reindex(list(self.handles.keys())).dropna()

        if exclude_singleton_clusters:
            counts = clusters["Clusters"].value_counts()
            singleton_clusters = counts[counts <= 1].index
            clusters = clusters[~clusters["Clusters"].isin(singleton_clusters)]

        self.clusters = clusters
        self.cluster_ids = clusters["Clusters"].unique().tolist()
        self.gauges_by_cluster = {
            c: clusters[clusters["Clusters"] == c].index.tolist() for c in self.cluster_ids
        }
        cluster_sizes = {c: len(g) for c, g in self.gauges_by_cluster.items()}
        total = sum(cluster_sizes.values())
        self.cluster_weights = {c: n / total for c, n in cluster_sizes.items()}

    def select_batch_gauges(self, n_gauges: int, rng: np.random.Generator):
        """Sample n_gauges gauges, proportionally to cluster size."""
        n_per_cluster = {c: max(1, round(w * n_gauges)) for c, w in self.cluster_weights.items()}
        selected = []
        for c, n in n_per_cluster.items():
            pool = self.gauges_by_cluster[c]
            n = min(n, len(pool))
            selected.extend(rng.choice(pool, size=n, replace=False).tolist())
        if len(selected) > n_gauges:
            selected = list(rng.choice(selected, size=n_gauges, replace=False))
        return selected

    def get_batch(self, gauge_ids, start_idx: int, seq_len: int):
        """{gauge_id: {dynamic, static, areas, obs, catchment_ids}} for a
        shared time window [start_idx, start_idx + seq_len)."""
        batch = {}
        for gid in gauge_ids:
            handle = self.handles[gid]
            dyn, obs = handle.get_window(start_idx, start_idx + seq_len)
            batch[gid] = {
                "dynamic": dyn,
                "static": handle.static,
                "areas": handle.areas,
                "obs": obs,
                "catchment_ids": handle.catchment_ids,
            }
        return batch
