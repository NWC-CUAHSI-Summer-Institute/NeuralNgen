"""
Per-gauge data handling and multi-gauge batch construction for training.

Rewritten to match src/training/train_multi_gauge.py's ACTUAL call sites
(MultiGaugeBatcher(gauge_ids=..., clusters=..., gauges_root=..., ...),
handle.date_to_idx(), handle.get_window_by_date(), batcher.get_batch(
n_gauges, batch_window_hours, start_date, end_date), batcher.cluster_to_gauges,
mutable batcher.gauge_ids/handles) rather than an earlier draft that used a
different constructor signature, a different get_batch signature, and assumed
pre-packaged hourly `QObs(mm/h)` CSVs instead of your actual daily USGS
NWIS `flow_m3s` parquet cache.

GaugeHandle       : lazy, per-gauge access to sub-catchment forcings
                    (NetCDF, hourly AORC), static physiographic attributes
                    and catchment areas (from the hydrofabric geopackage's
                    `divides`/`divide-attributes` tables), and observed
                    daily streamflow (from your QC'd usgs_streamflow_ref
                    parquet cache).
MultiGaugeBatcher : cluster-proportional gauge selection and batch
                    construction across many GaugeHandles.

IMPORTANT UNIT NOTE: streamflow here is `flow_m3s` (m^3/s), not the
`QObs(mm/h)` (specific discharge) the original CAMELS-based draft assumed.
Per-gauge z-score normalization (q_mean/q_std) works identically regardless
of units, so training itself is not broken by this - but the area-weighted
aggregation in train_multi_gauge.py implicitly assumes each catchment
contributes to the gauge total roughly in proportion to its area, which is
a much more direct physical statement in specific-discharge (mm/h) units
than in raw volumetric (m^3/s) units. This is the existing methodology's
assumption, not something changed here - flagging it because it's an
easy thing to lose track of when reading results.

MEMORY NOTE: like the earlier draft, this loads each gauge's full forcings
array into memory in GaugeHandle.__init__ (eager, not lazy). For one 25-year,
198-catchment, 8-variable test gauge this was ~164MB on disk. At 441 gauges,
even accounting for smaller catchment counts on many basins, this can easily
reach tens of GB resident in memory - calculate this against your actual
basin set before sizing a SLURM --mem request, or this will silently be
fine in early testing (few basins) and OOM only once you scale to the full
basin_file.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class GaugeHandle:
    """Lazy-loaded (at construction) per-gauge handle to forcings, static
    attributes, catchment areas, and observed daily streamflow."""

    def __init__(self, gauge_id: str, gauges_root, streamflow_dir,
                 dynamic_inputs, static_attributes):
        self.gauge_id = str(gauge_id)
        self.dynamic_inputs = list(dynamic_inputs)
        self.static_attributes = list(static_attributes)

        gauge_dir = Path(gauges_root) / f"gage-{self.gauge_id}"
        forcings_path = gauge_dir / "forcings" / "forcings.nc"
        if not forcings_path.exists():
            raise FileNotFoundError(f"No forcings.nc for gauge {self.gauge_id} at {forcings_path}")

        gpkg_matches = list((gauge_dir / "config").glob("*.gpkg"))
        if not gpkg_matches:
            raise FileNotFoundError(f"No subset gpkg found for gauge {self.gauge_id} under {gauge_dir}/config")
        gpkg_path = gpkg_matches[0]

        ds = xr.open_dataset(forcings_path)

        # This build of ngiab_data_cli's forcings.nc stores catchment-id and
        # time as bare dimensions with no coordinate index - the actual
        # catchment identifiers live in the `ids` data var (shape
        # (catchment-id,)), and the actual clock lives in the `Time` data var
        # (shape (catchment-id, time), Unix epoch seconds, int32) rather than
        # a `time` coordinate. Confirmed on real data: `Time` is identical
        # across every catchment row (one shared clock), so we take row 0.
        if "ids" in ds.data_vars:
            self.catchment_ids = [str(c) for c in ds["ids"].values]
        elif "catchment-id" in ds.coords:
            self.catchment_ids = [str(c) for c in ds["catchment-id"].values]
        elif "divide_id" in ds.coords:
            self.catchment_ids = [str(c) for c in ds["divide_id"].values]
        else:
            ds.close()
            raise KeyError(f"Gauge {self.gauge_id}: forcings.nc has none of "
                            f"'ids' (data var), 'catchment-id' or 'divide_id' "
                            f"(coords) to identify catchments")

        if "time" in ds.coords:
            time_vals = ds["time"].values
            self.time_index = pd.DatetimeIndex(pd.to_datetime(time_vals))
        elif "Time" in ds.data_vars:
            time_raw = ds["Time"].values
            if time_raw.ndim == 2:
                # (catchment-id, time) - verify every catchment really shares
                # one clock before assuming row 0 speaks for all of them.
                if not np.array_equal(time_raw, np.broadcast_to(time_raw[0], time_raw.shape)):
                    ds.close()
                    raise ValueError(
                        f"Gauge {self.gauge_id}: 'Time' values differ across "
                        f"catchments - can't assume a single shared clock"
                    )
                time_raw = time_raw[0]
            self.time_index = self._decode_epoch_seconds(time_raw)
        else:
            ds.close()
            raise KeyError(f"Gauge {self.gauge_id}: forcings.nc has neither a "
                            f"'time' coordinate nor a 'Time' data var")

        self.n_catchments = len(self.catchment_ids)

        missing = [v for v in self.dynamic_inputs if v not in ds.data_vars]
        if missing:
            ds.close()
            raise KeyError(f"Gauge {self.gauge_id}: forcings.nc missing dynamic inputs {missing}")

        # (n_catchments, n_time, n_dynamic)
        self._dyn = np.stack([ds[var].values for var in self.dynamic_inputs], axis=-1).astype(np.float32)
        ds.close()

        # --- static attributes + areas, straight from the subset gpkg ---
        conn = sqlite3.connect(str(gpkg_path))
        try:
            # divide_id (e.g. "cat-2454") is the catchment-id key used
            # throughout this pipeline - NOT the `id` column, which holds
            # flowpath-style "wb-XXXX" identifiers. Using `id` here silently
            # produces garbage area weights.
            area_df = pd.read_sql("SELECT divide_id, areasqkm FROM divides", conn).set_index("divide_id")

            available_cols = [row[1] for row in conn.execute("PRAGMA table_info('divide-attributes')").fetchall()]
            resolved_cols = self._resolve_attribute_columns(self.static_attributes, available_cols)

            attr_cols = ", ".join(["divide_id"] + [f'"{c}"' for c in resolved_cols.values()])
            static_df = pd.read_sql(f"SELECT {attr_cols} FROM 'divide-attributes'", conn).set_index("divide_id")
            static_df = static_df.rename(columns={actual: requested for requested, actual in resolved_cols.items()})
        finally:
            conn.close()

        self.areas = area_df.reindex(self.catchment_ids)["areasqkm"].values.astype(np.float32)
        self.static = static_df.reindex(self.catchment_ids)[self.static_attributes].values.astype(np.float32)

        if np.isnan(self.areas).any():
            raise ValueError(
                f"Gauge {self.gauge_id}: {int(np.isnan(self.areas).sum())} catchments have no area match - "
                f"divide_id mismatch between forcings.nc coordinates and the gpkg's divides table"
            )

        # --- observed daily streamflow: your QC'd usgs_streamflow_ref cache ---
        sf_path = Path(streamflow_dir) / f"{self.gauge_id}.parquet"
        if not sf_path.exists():
            raise FileNotFoundError(f"No streamflow parquet for gauge {self.gauge_id} at {sf_path}")
        sf = pd.read_parquet(sf_path)
        if not isinstance(sf.index, pd.DatetimeIndex):
            sf = sf.set_index("datetime")
        idx = pd.to_datetime(sf.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        self._daily_q = sf["flow_m3s"].set_axis(idx).sort_index()

        # Hourly-grid alias of the daily series: one real value per calendar
        # day, placed at that day's FIRST hourly timestamp in time_index, NaN
        # everywhere else. This exists so `.q_obs.values[start_idx:end_idx]`
        # can be sliced with the same hourly integer index date_to_idx
        # produces for `_dyn` - train_multi_gauge.py does exactly this for
        # its whole-train-period q_mean/q_std computation, relying on
        # np.nanmean/np.nanstd to skip the NaN padding. get_window_by_date()
        # below does its OWN direct daily lookup for per-batch windows -
        # it does not depend on this sparse alias.
        self.q_obs = self._build_hourly_sparse_q_alias()

    @staticmethod
    def _resolve_attribute_columns(requested: list, available_cols: list) -> dict:
        """Map YAML-style attribute names (hydrofabric model_attributes
        convention, e.g. 'mode.bexp_soil_layers_stag=1' - an aggregation
        method prefix, dotted, plus a soil-layer index after '=') onto the
        gpkg's actual 'divide-attributes' column names, which typically drop
        the aggregation prefix and use an underscore before the layer index
        (e.g. 'bexp_soil_layers_stag_1'). Tries a set of plausible rewrites
        per name, case-insensitively, and fails loudly - listing exactly
        which requested names couldn't be matched, plus every available
        column whose base name is a substring match - rather than silently
        guessing wrong or requiring a hand-built mapping table up front.
        Returns {requested_name: actual_column_name}."""
        available_lower = {c.lower(): c for c in available_cols}

        def candidates(name: str):
            seen, out = set(), []

            def add(c):
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)

            add(name)
            rest = name.split(".", 1)[1] if "." in name else name
            add(rest)
            add(rest.replace("=", "_"))
            add(name.replace("=", "_"))
            add(rest.replace("=", "."))
            add(rest.replace("=", ""))
            return out

        resolved, unresolved = {}, []
        for name in requested:
            match = None
            for cand in candidates(name):
                if cand.lower() in available_lower:
                    match = available_lower[cand.lower()]
                    break
            if match is None:
                unresolved.append(name)
            else:
                resolved[name] = match

        if unresolved:
            base_names = {n.split(".", 1)[-1].split("=", 1)[0].lower() for n in unresolved}
            near_matches = sorted(
                c for c in available_cols
                if any(base in c.lower() for base in base_names)
            )
            raise KeyError(
                "Could not resolve these static_attributes names to actual "
                "'divide-attributes' columns: " + ", ".join(unresolved) + ". "
                "Columns in this gpkg with a similar base name: " +
                (", ".join(near_matches) if near_matches else "(none found)") +
                ". Full column list: " + ", ".join(sorted(available_cols))
            )
        return resolved

    @staticmethod
    def _decode_epoch_seconds(raw) -> pd.DatetimeIndex:
        """Decode an int array of Unix epoch seconds into a DatetimeIndex,
        with a sanity check on the resulting year range - a silently wrong
        unit (e.g. ms instead of s) would otherwise produce dates so far off
        that every downstream date_to_idx/get_window_by_date call would
        quietly return garbage instead of failing loudly."""
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(raw), unit="s", utc=True)).tz_localize(None)
        if idx.min().year < 1990 or idx.max().year > 2035:
            raise ValueError(
                f"Decoded 'Time' range looks wrong ({idx.min()} to {idx.max()}) - "
                f"check the actual units of this file's Time variable"
            )
        return idx

    def _build_hourly_sparse_q_alias(self) -> pd.Series:
        day_labels = pd.Series(self.time_index.normalize(), index=np.arange(len(self.time_index)))
        first_pos_per_day = day_labels.groupby(day_labels).apply(lambda s: s.index[0])
        # first_pos_per_day: index=calendar day, value=position of that
        # day's first hourly timestamp within time_index
        daily_aligned = self._daily_q.reindex(first_pos_per_day.index)
        hourly_vals = np.full(len(self.time_index), np.nan, dtype=np.float32)
        hourly_vals[first_pos_per_day.values] = daily_aligned.values.astype(np.float32)
        return pd.Series(hourly_vals, index=self.time_index)

    def date_to_idx(self, date) -> int:
        """Hourly integer position of `date` within self.time_index.
        Clips to [0, len-1] if outside the record's actual coverage, rather
        than raising - callers are expected to check `end_idx - start_idx`
        against the window length before using the result."""
        date = pd.Timestamp(date)
        if date <= self.time_index[0]:
            return 0
        if date >= self.time_index[-1]:
            return len(self.time_index) - 1
        return int(self.time_index.searchsorted(date))

    def get_window_by_date(self, start_date, window_hours: int):
        """(dynamic, obs) for the window starting at start_date.
        dynamic shape: (n_catchments, window_hours, n_dynamic), hourly.
        obs shape: (window_hours // 24,), one value per calendar day in the
        window, NaN where that day is missing from the streamflow record.
        Returns (None, None) if the window would run past the end of the
        available forcings record."""
        start_date = pd.Timestamp(start_date)
        start_idx = self.date_to_idx(start_date)
        end_idx = start_idx + window_hours
        if end_idx > len(self.time_index):
            return None, None

        dyn = self._dyn[:, start_idx:end_idx, :]

        if window_hours % 24 != 0:
            raise ValueError(
                f"window_hours ({window_hours}) must be a multiple of 24 to align "
                f"hourly forcings with daily streamflow targets"
            )
        t_b = window_hours // 24
        daily_dates = pd.date_range(start_date.normalize(), periods=t_b, freq="D")
        obs = self._daily_q.reindex(daily_dates).values.astype(np.float32)
        return dyn, obs


class MultiGaugeBatcher:
    """Cluster-proportional gauge selection and batch construction across
    many GaugeHandles for multi-gauge training."""

    def __init__(self, gauge_ids, clusters: dict, gauges_root, streamflow_dir,
                 dynamic_inputs, static_attributes, seed: int = 42,
                 exclude_singleton_clusters: bool = False):
        self.gauge_ids = []
        self.handles = {}
        skipped = []
        for gid in tqdm(gauge_ids, desc="Loading gauge handles", unit="gauge"):
            try:
                self.handles[gid] = GaugeHandle(
                    gid, gauges_root, streamflow_dir, dynamic_inputs, static_attributes
                )
                self.gauge_ids.append(gid)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                skipped.append((gid, str(exc)))

        if skipped:
            print(f"MultiGaugeBatcher: skipped {len(skipped)} gauges with missing/invalid data:")
            for gid, err in skipped[:20]:
                print(f"  {gid}: {err}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")

        self.cluster_to_gauges = {}
        for gid in self.gauge_ids:
            c = clusters.get(gid)
            if c is None:
                continue
            self.cluster_to_gauges.setdefault(c, []).append(gid)

        if exclude_singleton_clusters:
            singleton = [c for c, g in self.cluster_to_gauges.items() if len(g) <= 1]
            for c in singleton:
                for gid in list(self.cluster_to_gauges[c]):
                    if gid in self.gauge_ids:
                        self.gauge_ids.remove(gid)
                    self.handles.pop(gid, None)
                del self.cluster_to_gauges[c]
            if singleton:
                print(f"Excluded {len(singleton)} singleton cluster(s): {singleton}")
        else:
            n_singleton = sum(1 for g in self.cluster_to_gauges.values() if len(g) <= 1)
            if n_singleton:
                print(f"WARNING: {n_singleton} cluster(s) have only 1 gauge - that gauge will be "
                      f"selected every time its cluster's proportional quota is >=1, every batch. "
                      f"Pass exclude_singleton_clusters=True to drop them instead.")

        self.rng = np.random.default_rng(seed)

    def _select_batch_gauges(self, n_gauges: int) -> list:
        cluster_ids = list(self.cluster_to_gauges.keys())
        if not cluster_ids:
            return []
        sizes = {c: len(g) for c, g in self.cluster_to_gauges.items()}
        total = sum(sizes.values())
        n_per_cluster = {c: max(1, round(sizes[c] / total * n_gauges)) for c in cluster_ids}

        selected = []
        for c, n in n_per_cluster.items():
            pool = self.cluster_to_gauges[c]
            n = min(n, len(pool))
            if n <= 0:
                continue
            picked = self.rng.choice(pool, size=n, replace=False)
            selected.extend(picked.tolist())

        if len(selected) > n_gauges:
            selected = list(self.rng.choice(selected, size=n_gauges, replace=False))
        return selected

    def get_batch(self, n_gauges: int, batch_window_hours: int, train_start_date, train_end_date) -> dict:
        """Select ~n_gauges gauges (cluster-proportional) and, for each, a
        random valid hourly window in [train_start_date, train_end_date).
        Gauges with insufficient room for a full window, or whose window
        would run past the record's end, are silently dropped from this
        batch (not padded, not errored) - callers should expect
        `len(selected_gauges) <= n_gauges`, sometimes 0."""
        selected = self._select_batch_gauges(n_gauges)

        per_gauge_dyn, per_gauge_static, per_gauge_qobs, per_gauge_areas = {}, {}, {}, {}
        actually_selected = []

        for gid in selected:
            handle = self.handles[gid]
            start_idx = handle.date_to_idx(train_start_date)
            end_idx = handle.date_to_idx(train_end_date)
            if end_idx - start_idx <= batch_window_hours:
                continue

            start_row = self.rng.integers(start_idx, end_idx - batch_window_hours)
            start_date = handle.time_index[start_row]

            dyn, obs = handle.get_window_by_date(start_date, batch_window_hours)
            if dyn is None:
                continue

            per_gauge_dyn[gid] = dyn
            per_gauge_static[gid] = handle.static
            per_gauge_qobs[gid] = obs
            per_gauge_areas[gid] = handle.areas
            actually_selected.append(gid)

        return {
            "selected_gauges": actually_selected,
            "per_gauge_dyn": per_gauge_dyn,
            "per_gauge_static": per_gauge_static,
            "per_gauge_qobs": per_gauge_qobs,
            "per_gauge_areas": per_gauge_areas,
        }