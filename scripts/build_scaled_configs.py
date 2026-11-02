"""
For every gauge x scenario combination in CAMELS_US_ngiab_full (which has
forcings.nc but NO pre-existing troute.yaml/outputs/):

  1. Generate troute_{scenario}.yaml from the ONE known-working template,
     substituting this gauge's own geo_file_path, corrected date range, and
     scenario-specific I/O paths.
  2. Generate realization_{scenario}.json (explicit per-catchment).
  3. Copy this scenario's bmi configs into the gauge's own folder.
  4. Create outputs/ngen_{scenario}/, outputs/troute_{scenario}/, and
     outputs/troute_{scenario}_stream/ dirs.

FIXED vs the first version of this script: build_troute_yaml_from_template()
originally used regexes assuming single-space "key: value" formatting and a
space-separated start_datetime ("2020-01-01 00:00:00"). The actual template
in production (gauge_01073000/routing_config.yaml) column-aligns its keys
with many spaces before the colon ("nts                         : 8640"),
quotes start_datetime with an UNDERSCORE separator ("2015-12-01_00:00:00"),
and has no "stream_output_directory" key at all - it uses the older
chanobs_output block (chanobs_output_directory) instead. The original
regexes silently matched zero times against all of that (re.sub with no
matches just returns the string unchanged, no error), so nts, max_loop_size,
qlat_input_folder, start_datetime, and the output directory were NEVER
actually being customized per gauge/scenario - every generated
troute_{scenario}.yaml would have silently kept this one test gauge's
original values. Fixed here to:
  - match keys with ANY amount of whitespace before the colon (\\s*:\\s*)
  - write start_datetime back in the SAME quoted-underscore format the
    template uses, not a space-separated one
  - target chanobs_output_directory (the key this template actually has)
    instead of a stream_output_directory key that doesn't exist in it

FIXED AGAIN (schema mismatch, confirmed via an actual ngen-serial run on
gauge 01013500): the template's column mappings for `dx` and `waterbody`
point at column names from an OLDER hydrofabric schema that this
production data doesn't have (`length_m` vs the real `Length_m`;
`rl_NHDWaterbodyComID` vs the real `WaterbodyID`). These crashes only show
up once t-route actually runs routing - not during config generation, and
not during the LSTM/BMI stage, which completes fine first. Also stopped
substituting level_pool_waterbody_parameter_file_path with a per-gauge
path and now comment it out instead - reservoir/lake level-pool routing is
disabled for every gauge anyway (break_network_at_waterbodies: False), so
leaving that key pointed at a gpkg is unnecessary risk. See
build_troute_yaml_from_template() for the full explanation.

FIXED AGAIN (CHANOBS always empty + qlat parquet cache dumped at gauge
root, both confirmed via actual runs on gauges 01013500 and 01117468):
  - CHANOBS.nc writes successfully but ALWAYS has an empty feature_id
    array, with no error, because t-route's HYFeaturesNetwork.read_geopkg()
    only loads the 'network' gpkg layer (which holds the real gauge
    crosswalk, hl_uri) when one of the data-assimilation flags
    (streamflow_nudging / reservoir_persistence_usgs / etc.) is True. Every
    gauge here runs with all of those False (plain retrospective
    simulation, no real-time USGS nudging), so that layer never loads and
    CHANOBS's crosswalk is permanently empty. Turning streamflow_nudging
    on to work around it just trades this for a hard crash in
    DataAssimilation.py's _create_usgs_df (it unconditionally tries to
    build a Path from usgs_timeslices_folder, which is None/unset for us).
    Fix: add a 'stream_output' block instead. It's t-route's newer output
    mechanism, writes routed flow/velocity/depth for EVERY feature_id in
    the network unconditionally (no DA/crosswalk dependency at all), and
    is what the evaluation pipeline should read instead of CHANOBS.
    Verified end-to-end on gauge 01013500: feature_id=5539 (that gauge's
    outlet, wb-5539) present with real, varying flow values once this
    block was added - permanently empty before, every single time.
  - t-route's qlat reformatting step (triggered by
    qlat_file_pattern_filter="nex-*", which this template already sets)
    writes one NEXOUT.parquet file per catchment per hour of simulation
    into whatever 'binary_nexus_file_folder' says - and the template's own
    default for that key is './', i.e. the gauge's own root directory.
    Confirmed on gauge 01117468: ~745 loose .parquet files dumped directly
    into gage-01117468/ on every run. Fixed by pointing this at
    '/ngen/qlat_cache/' instead - a fixed container-side path that
    run_ngen_scenario.py now bind-mounts to a per-gauge/scenario folder on
    /scratch (wiped and recreated before every run, since t-route refuses
    to start if this folder already has leftover .parquet files in it).
"""
import json
import re
import shutil
import sys
from pathlib import Path

VARIABLES_NAMES_MAP = {
    "atmosphere_water__liquid_equivalent_precipitation_rate": "APCP_surface",
    "land_surface_radiation~incoming~longwave__energy_flux": "DLWRF_surface",
    "land_surface_radiation~incoming~shortwave__energy_flux": "DSWRF_surface",
    "land_surface_air__pressure": "PRES_surface",
    "atmosphere_air_water~vapor__relative_saturation": "SPFH_2maboveground",
    "land_surface_air__temperature": "TMP_2maboveground",
    "land_surface_wind__x_component_of_velocity": "UGRD_10maboveground",
    "land_surface_wind__y_component_of_velocity": "VGRD_10maboveground",
}

START_TIME = "2018-01-01 00:00:00"
END_TIME = "2023-02-01 00:00:00"
TROUTE_START = "2018-01-01_00:00:00"
TROUTE_NTS = 534816


def build_catchment_formulation(config_rel_path: str) -> dict:
    return {
        "formulations": [
            {
                "name": "bmi_python",
                "params": {
                    "python_type": "src.models.bmi_catchmentlstm.bmi_CatchmentLSTM",
                    "model_type_name": "bmi_CatchmentLSTM",
                    "init_config": config_rel_path,
                    "main_output_variable": "land_surface_water__runoff_volume_flux",
                    "allow_exceed_end_time": True,
                    "uses_forcing_file": False,
                    "variables_names_map": VARIABLES_NAMES_MAP,
                }
            }
        ],
        "forcing": {
            "path": "./forcings/forcings.nc",
            "provider": "NetCDF",
            "enable_cache": False,
        }
    }


def build_realization(bmi_config_dir: Path, scenario: str, troute_filename: str) -> dict:
    yml_files = sorted(bmi_config_dir.glob("*.yml"))
    catchments = {}
    for yml_file in yml_files:
        catchment_id = yml_file.stem
        rel_path = f"./bmi_config_files_{scenario}/{yml_file.name}"
        catchments[catchment_id] = build_catchment_formulation(rel_path)

    if not catchments:
        return None

    first_init_config = next(iter(catchments.values()))["formulations"][0]["params"]["init_config"]

    return {
        "global": {
            "formulations": [
                {
                    "name": "bmi_python",
                    "params": {
                        "python_type": "src.models.bmi_catchmentlstm.bmi_CatchmentLSTM",
                        "model_type_name": "bmi_CatchmentLSTM",
                        "init_config": first_init_config,
                        "main_output_variable": "land_surface_water__runoff_volume_flux",
                        "allow_exceed_end_time": True,
                        "uses_forcing_file": False,
                        "variables_names_map": VARIABLES_NAMES_MAP,
                    }
                }
            ],
            "forcing": {
                "path": "./forcings/forcings.nc",
                "provider": "NetCDF",
                "enable_cache": False,
            }
        },
        "catchments": catchments,
        "time": {
            "start_time": START_TIME,
            "end_time": END_TIME,
            "output_interval": 3600,
        },
        "routing": {
            "t_route_config_file_with_path": f"./config/{scenario}/{troute_filename}"
        },
        "output_root": f"./outputs/ngen_{scenario}",
    }


def _sub_key(pattern_key: str, new_value: str, text: str, quote: bool = False) -> tuple[str, int]:
    """Replace `<pattern_key>:<any whitespace><value>` with
    `<pattern_key>: <new_value>`, tolerating any amount of whitespace (or
    none) around the colon - matches column-aligned YAML like
    "nts                         : 8640" as well as compact "nts: 8640".
    Returns (new_text, n_substitutions) so callers can verify the key was
    actually found rather than silently doing nothing."""
    value_re = r'"[^"]*"|\S+'
    # \b anchors are essential here - without them "nts" matches inside
    # "synthetic_wb_segments:" (literally contains the substring "nts:"),
    # silently corrupting unrelated lines. Confirmed by testing against the
    # real template before this fix was added.
    pattern = re.compile(rf"(\b{re.escape(pattern_key)}\b\s*:\s*)({value_re})")
    replacement = f'"{new_value}"' if quote else new_value
    new_text, n = pattern.subn(lambda m: m.group(1) + replacement, text)
    return new_text, n


def _comment_out_key(pattern_key: str, text: str) -> tuple[str, int]:
    """Comment out an entire `<pattern_key>: <value>` line, any leading
    whitespace. Already-commented lines are left alone (0 substitutions),
    so this is safe to call even if the template's own copy of the key was
    already disabled."""
    pattern = re.compile(
        rf"^([ \t]*)(\b{re.escape(pattern_key)}\b\s*:\s*\S.*)$",
        re.MULTILINE,
    )
    return pattern.subn(lambda m: f"{m.group(1)}#{m.group(2)}", text)


def _insert_after_key(pattern_key: str, block_text: str, text: str) -> tuple[str, int]:
    """Insert `block_text` (its own trailing newline included, no leading
    one) immediately after the first line that sets `<pattern_key>:`, at
    any indentation. For adding a config block the template doesn't have
    at all (so there's no existing key for _sub_key to substitute into).
    Returns (new_text, n_insertions) - same "verify the anchor was
    actually found, never fail silently" philosophy as _sub_key() and
    _comment_out_key() above."""
    pattern = re.compile(
        rf"^([ \t]*\b{re.escape(pattern_key)}\b\s*:.*)$",
        re.MULTILINE,
    )
    return pattern.subn(lambda m: m.group(1) + "\n" + block_text, text, count=1)


def build_troute_yaml_from_template(template_text: str, gauge_id: str, scenario: str) -> str:
    src = template_text
    new_gpkg = f"./config/gage-{gauge_id}_subset.gpkg"

    # Per-gauge value substitutions.
    substitutions = [
        ("geo_file_path", new_gpkg, False),
        ("start_datetime", TROUTE_START, True),
        ("nts", str(TROUTE_NTS), False),
        ("max_loop_size", str(TROUTE_NTS), False),
        ("qlat_input_folder", f"./outputs/ngen_{scenario}/", False),
        ("chanobs_output_directory", f"./outputs/troute_{scenario}/", False),
        # Schema fixes - static across every gauge, not per-gauge values.
        # The template (gauge_01073000/routing_config.yaml) was written
        # against an OLDER hydrofabric schema than what's actually in
        # production (CAMELS_US_ngiab_full/*/config/*.gpkg). Confirmed via
        # `sqlite3 <gpkg> "PRAGMA table_info('flowpath-attributes')"` on
        # gauge 01013500's gpkg directly:
        #   - the real column is 'Length_m' (capital L), not the template's
        #     lowercase 'length_m'. This mismatch is INVISIBLE until t-route
        #     actually runs routing - the LSTM/BMI stage runs and finishes
        #     fine first, then compute_nhd_routing_v02 dies with
        #     KeyError: "['dx'] not in index" reading the mis-mapped column.
        #   - the real column is 'WaterbodyID', not the template's NWM-era
        #     'rl_NHDWaterbodyComID', which doesn't exist in this hydrofabric
        #     at all. t-route's HYFeaturesNetwork.preprocess_waterbodies
        #     unconditionally runs `self.dataframe.drop('waterbody', axis=1)`
        #     at the end of that function regardless of
        #     break_network_at_waterbodies, so a 'waterbody' column must
        #     exist post-rename even when waterbody/reservoir routing itself
        #     is off, or it's a hard crash (KeyError: "['waterbody'] not
        #     found in axis") before any routing even starts.
        # Verified end-to-end: gauge 01013500 ran clean (745 LSTM timesteps
        # + full t-route routing + CHANOBS output written) only after both
        # of these were corrected.
        ("dx", "'Length_m'", False),
        ("waterbody", "'WaterbodyID'", False),
        ("gages", "'gage'", False),
        # binary_nexus_file_folder: the template's own default is './' -
        # t-route's qlat reformatting step (triggered by
        # qlat_file_pattern_filter="nex-*", set below) writes one
        # NEXOUT.parquet file per catchment per simulated hour into
        # whatever this says, so './' means ~745 loose files dumped
        # straight into the gauge's root directory on every run.
        # '/ngen/qlat_cache/' is a fixed container-side path that
        # run_ngen_scenario.py bind-mounts to a per-gauge/scenario folder
        # on /scratch instead (wiped clean before every run - t-route
        # refuses to start if this folder isn't empty).
        ("binary_nexus_file_folder", "/ngen/qlat_cache/", False),
    ]

    missing = []
    for key, value, quote in substitutions:
        src, n = _sub_key(key, value, src, quote=quote)
        if n == 0:
            missing.append(key)

    if missing:
        raise ValueError(
            f"troute template is missing expected key(s), nothing was substituted "
            f"for: {missing}. Check the template still has these keys (possibly "
            f"under a different name in a newer/older t-route schema) before "
            f"trusting the generated troute_{scenario}.yaml files."
        )

    # level_pool_waterbody_parameter_file_path is commented out, NOT
    # substituted with a per-gauge path (the previous version of this
    # script did that). Reservoir/lake level-pool routing is disabled for
    # every gauge (break_network_at_waterbodies: False in the template), so
    # this key is unused - leaving it pointed at a gpkg whose lake-table
    # schema doesn't match what t-route expects is unnecessary crash risk
    # across hundreds of gauges for a feature that's off anyway.
    src, _ = _comment_out_key("level_pool_waterbody_parameter_file_path", src)

    # stream_output: the template predates this t-route feature entirely -
    # it only has chrtout_output/chanobs_output, so there's no existing
    # key for _sub_key() to substitute into. Insert a brand-new block
    # instead, anchored right after lakeout_output so it lands inside
    # output_parameters at the correct indentation.
    #
    # Why this exists instead of just fixing CHANOBS: CHANOBS only gets
    # the real gauge crosswalk loaded when a data-assimilation flag is
    # True (see module docstring) - with every DA flag False here, it
    # silently writes an empty feature_id array forever, no error.
    # stream_output writes flow/velocity/depth for EVERY feature_id in the
    # network unconditionally, so it's immune to that gate entirely.
    # stream_output_time: -1 means "one file for the whole run" instead of
    # the default 1-file-per-simulated-hour (~745 small files per
    # gauge/scenario otherwise - unnecessary clutter across 500+ gauges).
    stream_output_block = (
        "    stream_output:\n"
        f"        stream_output_directory: ./outputs/troute_{scenario}_stream/\n"
        "        stream_output_time: -1\n"
    )
    src, n_stream = _insert_after_key("lakeout_output", stream_output_block, src)
    if n_stream == 0:
        raise ValueError(
            f"troute template is missing 'lakeout_output' (used as the anchor "
            f"to insert the stream_output block) - nothing was inserted for "
            f"troute_{scenario}.yaml. Check the template's output_parameters "
            f"section before trusting the generated file."
        )

    return src


def process_gauge(gauge_dir: Path, bmi_config_root: Path, scenarios: list, troute_template: str):
    gauge_id = gauge_dir.name.replace("gage-", "")
    config_dir = gauge_dir / "config"
    if not config_dir.exists():
        return False, "no config dir"

    gpkg_matches = list(config_dir.glob("*_subset.gpkg"))
    if not gpkg_matches:
        return False, "no subset gpkg"

    any_scenario_written = False
    for scenario in scenarios:
        bmi_dir = bmi_config_root / scenario / gauge_id
        if not bmi_dir.exists():
            continue

        realization = build_realization(bmi_dir, scenario, f"troute_{scenario}.yaml")
        if realization is None:
            continue

        (config_dir / f"realization_{scenario}.json").write_text(json.dumps(realization, indent=2))

        scenario_config_dir = config_dir / scenario
        scenario_config_dir.mkdir(parents=True, exist_ok=True)
        troute_content = build_troute_yaml_from_template(troute_template, gauge_id, scenario)
        (scenario_config_dir / f"troute_{scenario}.yaml").write_text(troute_content)

        dest = gauge_dir / f"bmi_config_files_{scenario}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(bmi_dir, dest)

        (gauge_dir / "outputs" / f"ngen_{scenario}").mkdir(parents=True, exist_ok=True)
        (gauge_dir / "outputs" / f"troute_{scenario}").mkdir(parents=True, exist_ok=True)
        (gauge_dir / "outputs" / f"troute_{scenario}_stream").mkdir(parents=True, exist_ok=True)

        any_scenario_written = True

    return any_scenario_written, "ok" if any_scenario_written else "no matching bmi configs"


if __name__ == "__main__":
    gauges_root = Path(sys.argv[1])
    bmi_config_root = Path(sys.argv[2])
    scenarios = sys.argv[3].split(",")
    troute_template_path = Path(sys.argv[4])

    troute_template = troute_template_path.read_text()

    gauge_dirs = sorted(gauges_root.glob("gage-*"))
    print(f"Processing {len(gauge_dirs)} gauge directories for scenarios: {scenarios}")

    n_ok, n_skip = 0, 0
    skip_reasons = {}
    for i, gauge_dir in enumerate(gauge_dirs):
        ok, msg = process_gauge(gauge_dir, bmi_config_root, scenarios, troute_template)
        if ok:
            n_ok += 1
        else:
            n_skip += 1
            skip_reasons[msg] = skip_reasons.get(msg, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gauge_dirs)}] processed ({n_ok} ok, {n_skip} skipped)")

    print(f"\nDone. {n_ok} gauges processed, {n_skip} skipped.")
    if skip_reasons:
        print("Skip reasons:", skip_reasons)