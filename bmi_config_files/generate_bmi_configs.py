"""
Generate one BMI init-config YAML per catchment for ONE gauge, pulling
areas and static attributes straight from that gauge's own subset .gpkg -
the exact same source multi_gauge_batcher.py's GaugeHandle reads at
training time (divides.areasqkm, divide-attributes columns matched to the
checkpoint's own static_attributes list). No separate hand-built
"bmi_config_root" dataset is needed - this generates it.

Writes into <bmi_config_root>/<scenario>/<gauge_id>/<catchment_id>.yml,
exactly the layout build_scaled_configs.py's process_gauge() expects
(bmi_config_root / scenario / gauge_id / *.yml), so you can run
build_scaled_configs.py unmodified afterward.

Targets the lstm.base.BmiBase-style wrapper (src/models/bmi_catchmentlstm.py)
- the one build_scaled_configs.py's python_type already points at - which
reads: model_pt, gauge_id, area_sqkm, static_attribute_values (an ordered
list matching the checkpoint's static_attributes order).

model_pt is written as a CONTAINER path (/ngen/NeuralNGEN/...), not the
host path, because run_ngen_scenario.py bind-mounts neuralngen_root to
/ngen/NeuralNGEN inside the apptainer container - an absolute host path
here would not exist inside the container and initialize() would fail
with FileNotFoundError on every catchment.

Usage:
    python generate_bmi_configs_for_gauge.py \\
        <gauge_id> <scenario> <checkpoint_path> \\
        <gauges_root> <neuralngen_root> <bmi_config_root>

Example:
    python generate_bmi_configs_for_gauge.py \\
        01013500 rfl \\
        /bighome/hdagne1/Docker/NeuralNGEN/runs/scoring_rule_multi_rfl/model_final.pt \\
        /bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full \\
        /bighome/hdagne1/Docker/NeuralNGEN \\
        /bighome/hdagne1/Docker/NeuralNGEN/bmi_config_files
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from ruamel.yaml import YAML


def resolve_attribute_columns(requested: list, available_cols: list) -> dict:
    """Same name-resolution logic as multi_gauge_batcher.py's GaugeHandle -
    must match EXACTLY, or the static attribute values fed to ngen at
    inference time won't be the same physical quantities the model was
    trained on for this catchment."""
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


def main():
    gauge_id = sys.argv[1]
    scenario = sys.argv[2]
    checkpoint_path = Path(sys.argv[3])
    gauges_root = Path(sys.argv[4])
    neuralngen_root = Path(sys.argv[5])
    bmi_config_root = Path(sys.argv[6])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cfg = ckpt["config"]
    static_attributes = train_cfg["static_attributes"]

    if gauge_id not in ckpt["q_stats"]:
        raise KeyError(
            f"gauge_id '{gauge_id}' not in this checkpoint's trained gauges "
            f"({len(ckpt['q_stats'])} available). This checkpoint cannot "
            f"produce a de-normalization scale for a gauge it never saw in training."
        )

    # model_pt as the CONTAINER path - checkpoint_path must actually be
    # under neuralngen_root for this rewrite to make sense.
    try:
        rel_ckpt = checkpoint_path.resolve().relative_to(neuralngen_root.resolve())
    except ValueError:
        raise ValueError(
            f"checkpoint_path ({checkpoint_path}) is not under neuralngen_root "
            f"({neuralngen_root}) - it needs to be, so it's reachable at "
            f"/ngen/NeuralNGEN/... inside the apptainer container."
        )
    model_pt_container_path = f"/ngen/NeuralNGEN/{rel_ckpt.as_posix()}"

    gauge_dir = gauges_root / f"gage-{gauge_id}"
    forcings_path = gauge_dir / "forcings" / "forcings.nc"
    gpkg_matches = list((gauge_dir / "config").glob("*.gpkg"))
    if not gpkg_matches:
        raise FileNotFoundError(f"No subset gpkg under {gauge_dir}/config")
    gpkg_path = gpkg_matches[0]

    ds = xr.open_dataset(forcings_path)
    if "ids" in ds.data_vars:
        catchment_ids = [str(c) for c in ds["ids"].values]
    elif "catchment-id" in ds.coords:
        catchment_ids = [str(c) for c in ds["catchment-id"].values]
    elif "divide_id" in ds.coords:
        catchment_ids = [str(c) for c in ds["divide_id"].values]
    else:
        ds.close()
        raise KeyError(f"forcings.nc for gauge {gauge_id} has no catchment-id source")
    ds.close()

    conn = sqlite3.connect(str(gpkg_path))
    try:
        import pandas as pd
        area_df = pd.read_sql("SELECT divide_id, areasqkm FROM divides", conn).set_index("divide_id")
        available_cols = [row[1] for row in conn.execute("PRAGMA table_info('divide-attributes')").fetchall()]
        resolved_cols = resolve_attribute_columns(static_attributes, available_cols)
        attr_cols = ", ".join(["divide_id"] + [f'"{c}"' for c in resolved_cols.values()])
        static_df = pd.read_sql(f"SELECT {attr_cols} FROM 'divide-attributes'", conn).set_index("divide_id")
        static_df = static_df.rename(columns={actual: requested for requested, actual in resolved_cols.items()})
    finally:
        conn.close()

    areas = area_df.reindex(catchment_ids)["areasqkm"]
    static = static_df.reindex(catchment_ids)[static_attributes]

    # Whole-basin area (all divides): the model is trained on gauge-scale m3/s with an
    # area-weighted mean, so the BMI wrapper needs this to convert to ngen's depth contract.
    gauge_area_sqkm = float(area_df["areasqkm"].sum())

    missing_area = areas[areas.isna()].index.tolist()
    if missing_area:
        raise ValueError(f"Gauge {gauge_id}: catchments with no area match in gpkg divides table: {missing_area}")
    missing_static = static[static.isna().any(axis=1)].index.tolist()
    if missing_static:
        raise ValueError(f"Gauge {gauge_id}: catchments with missing static attribute values: {missing_static}")

    out_dir = bmi_config_root / scenario / gauge_id
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml = YAML()
    yaml.default_flow_style = False

    n_written = 0
    for cat_id in catchment_ids:
        cfg = {
            "train_cfg_file": model_pt_container_path,
            "gauge_id": gauge_id,
            "area_sqkm": float(areas.loc[cat_id]),
            "gauge_area_sqkm": gauge_area_sqkm,
            "static_attributes": {
                name: float(v) for name, v in zip(static_attributes, static.loc[cat_id].values)
            },
        }
        out_path = out_dir / f"{cat_id}.yml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f)
        n_written += 1

    print(f"Wrote {n_written} per-catchment BMI configs to {out_dir}")
    print(f"model_pt (container path, verify this exists once inside apptainer): {model_pt_container_path}")
    print(f"static_attributes order used ({len(static_attributes)}): {static_attributes}")


if __name__ == "__main__":
    main()