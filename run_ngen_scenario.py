"""
Run one (gauge_id, scenario) combination's ngen-serial simulation +
t-route routing inside the CIROH NextGen apptainer image.

This is the script one SLURM array task calls, per line of
manifests/ngen_run_manifest.txt ("gauge_id,scenario"). It did NOT exist
anywhere on disk before this - it's referenced by comments in
generate_manisfest.py, generate_bmi_configs.py and ngen_run_config.yml,
and by the (also-missing) original array job script, but was never
actually written or saved. This is a fresh implementation, built directly
from the exact apptainer invocation verified end-to-end (745 LSTM
timesteps + full t-route routing + CHANOBS output) on gauge 01013500
during today's debugging - not a guess.

Usage:
    python run_ngen_scenario.py <config_yml_path> <gauge_id> <scenario>

Example:
    python run_ngen_scenario.py \\
        /bighome/hdagne1/Docker/NeuralNGEN/ngen_run_config.yml \\
        01013500 rfl

Exits non-zero (propagating ngen-serial's own exit code where possible)
on any failure, so the calling sbatch array task shows as FAILED in
squeue/sacct rather than silently succeeding.

UPDATE (stream_output / qlat_binary_cache fix): t-route's qlat reformatting
step (triggered by qlat_file_pattern_filter="nex-*") writes per-catchment
NEXOUT.parquet cache files into whatever "binary_nexus_file_folder" the
troute YAML specifies. That folder must (a) exist and (b) be empty on
every run, or t-route raises a RuntimeError. To avoid dumping thousands
of these files into /bighome (limited quota) across 500+ gauges, this
script now wipes/recreates a per-gauge/scenario cache folder on /scratch
before every run and bind-mounts it into the container at a fixed path
(/ngen/qlat_cache). The troute YAML's binary_nexus_file_folder should be
set to "/ngen/qlat_cache/" to match (see build_scaled_configs.py).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <config_yml_path> <gauge_id> <scenario>",
              file=sys.stderr)
        sys.exit(2)

    config_path = Path(sys.argv[1])
    gauge_id = sys.argv[2]
    scenario = sys.argv[3]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gauges_root = Path(cfg["gauges_root"])
    neuralngen_root = Path(cfg["neuralngen_root"])
    ngen_sif = Path(cfg["ngen_sif"])
    qlat_cache_root = Path(cfg.get("qlat_cache_root", "/scratch/hdagne1/qlat_cache"))

    gauge_dir = gauges_root / f"gage-{gauge_id}"
    gpkg_rel = f"./config/gage-{gauge_id}_subset.gpkg"
    realization_rel = f"./config/realization_{scenario}.json"

    if not gauge_dir.exists():
        print(f"ERROR: gauge directory not found: {gauge_dir}", file=sys.stderr)
        sys.exit(1)
    if not (gauge_dir / "config" / f"gage-{gauge_id}_subset.gpkg").exists():
        print(f"ERROR: subset gpkg not found under {gauge_dir}/config/", file=sys.stderr)
        sys.exit(1)
    if not (gauge_dir / "config" / f"realization_{scenario}.json").exists():
        print(f"ERROR: realization_{scenario}.json not found under {gauge_dir}/config/ "
              f"- did build_scaled_configs.py actually generate this scenario for "
              f"this gauge? (it skips gauges with no matching bmi configs)", file=sys.stderr)
        sys.exit(1)
    if not ngen_sif.exists():
        print(f"ERROR: ngen_sif not found: {ngen_sif} - check ngen_run_config.yml",
              file=sys.stderr)
        sys.exit(1)

    # t-route's qlat binary-cache step refuses to run if its output folder
    # already contains leftover .parquet files from a prior run, so wipe
    # and recreate this gauge/scenario's scratch cache folder fresh every time.
    qlat_cache_dir = qlat_cache_root / f"gage-{gauge_id}" / scenario
    if qlat_cache_dir.exists():
        shutil.rmtree(qlat_cache_dir)
    qlat_cache_dir.mkdir(parents=True, exist_ok=True)

    inner_cmd = (
        f"cd /ngen/ngen/data && "
        f"/dmod/bin/ngen-serial {gpkg_rel} all {gpkg_rel} all {realization_rel}"
    )

    cmd = [
        "apptainer", "exec",
        "-B", f"{gauge_dir}:/ngen/ngen/data",
        "-B", f"{neuralngen_root}:/ngen/NeuralNGEN",
        "-B", f"{qlat_cache_dir}:/ngen/qlat_cache",
        "--env", "PYTHONPATH=/ngen/NeuralNGEN",
        str(ngen_sif),
        "bash", "-c", inner_cmd,
    ]

    print(f"gauge={gauge_id} scenario={scenario}")
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"gauge={gauge_id} scenario={scenario} FAILED "
              f"(ngen-serial exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"gauge={gauge_id} scenario={scenario} completed successfully.")


if __name__ == "__main__":
    main()