"""
RECONSTRUCTED (like catchment_lstm.py) - not recovered from the purge.
Built to satisfy the exact contract bmi_catchmentlstm.py calls:
    crosswalk.dynamic_input_vars      -> list[(name, unit)] for build_state()
    crosswalk.output_vars             -> list[(name, unit)] for build_state()
    crosswalk.crosswalk_to_external(internal_name) -> external BMI name

"external" = BMI standard names (the KEYS of VARIABLES_NAMES_MAP in
scripts/build_scaled_configs.py). ngen's forcing engine calls set_value()
on this component using these names.

"internal" = the raw AORC/forcing variable names this checkpoint was
trained on (train_cfg["dynamic_inputs"], e.g. "APCP_surface") - the VALUES
of VARIABLES_NAMES_MAP.

VERIFY the units below against your real crosswalk.py / AORC file metadata
before relying on get_var_units() output - they don't affect numerical
correctness of update() (that only depends on dict keys matching), only
BMI introspection metadata. If you still have the original crosswalk.py
anywhere, use that instead of this reconstruction.
"""
import typing

output_vars: list[tuple[str, str]] = [
    ("land_surface_water__runoff_depth", "m"),
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
]

# external BMI standard name -> (internal/training name, unit)
_CROSSWALK: dict[str, tuple[str, str]] = {
    "atmosphere_water__liquid_equivalent_precipitation_rate": ("APCP_surface", "kg m-2"),
    "land_surface_radiation~incoming~longwave__energy_flux": ("DLWRF_surface", "W m-2"),
    "land_surface_radiation~incoming~shortwave__energy_flux": ("DSWRF_surface", "W m-2"),
    "land_surface_air__pressure": ("PRES_surface", "Pa"),
    "atmosphere_air_water~vapor__relative_saturation": ("SPFH_2maboveground", "kg kg-1"),
    "land_surface_air__temperature": ("TMP_2maboveground", "K"),
    "land_surface_wind__x_component_of_velocity": ("UGRD_10maboveground", "m s-1"),
    "land_surface_wind__y_component_of_velocity": ("VGRD_10maboveground", "m s-1"),
}

dynamic_input_vars: list[tuple[str, str]] = [
    (external_name, unit) for external_name, (_, unit) in _CROSSWALK.items()
]

_INTERNAL_TO_EXTERNAL: dict[str, str] = {
    internal_name: external_name
    for external_name, (internal_name, _) in _CROSSWALK.items()
}


def crosswalk_to_external(internal_name: str) -> str:
    """Training/forcing-file variable name (e.g. 'APCP_surface') -> the BMI
    standard name ngen actually calls set_value() with for that variable."""
    try:
        return _INTERNAL_TO_EXTERNAL[internal_name]
    except KeyError:
        raise KeyError(
            f"'{internal_name}' has no entry in the crosswalk - it isn't one "
            f"of the dynamic inputs this BMI module knows how to receive. "
            f"Known internal names: {sorted(_INTERNAL_TO_EXTERNAL)}"
        )